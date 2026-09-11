"""
Step 3 — Embedding models.

Two models behind one interface, which is what makes Experiment 1 a fair fight:

  A. BAAI/bge-base-en-v1.5     (open-source, 768-dim, local)
  B. gemini-embedding-001      (commercial, 3072-dim, API)
  C. gemini-embedding-001@768  (same model truncated to bge's width)

Arm C is the interesting one. Comparing A against B confounds two variables --
model quality AND vector width. Arm C holds width constant so the comparison
isolates model quality. That distinction is a strong viva talking point.

Everything is wrapped in CacheBackedEmbeddings so you never pay to embed the
same chunk twice across notebook re-runs.
"""

from __future__ import annotations

import re
import time
from collections import deque

from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.storage import LocalFileStore

from . import config


# --------------------------------------------------------------------------
# Rate limiting for the Gemini free tier
# --------------------------------------------------------------------------

class RateLimiter:
    """
    Rolling-window limiter over the number of ITEMS sent per minute.

    This is the shape the Gemini quota actually takes. The free tier allows
    ~100 embed_content requests per minute per model, and batch_embed_contents
    with N texts consumes N of them — so limiting the number of API CALLS does
    nothing. We track the timestamp of every text sent and block until the
    oldest one falls outside the 60-second window.
    """

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._sent: deque[float] = deque()

    def acquire(self, n: int) -> None:
        n = min(n, self.max_per_minute)
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= 60.0:
                self._sent.popleft()

            if len(self._sent) + n <= self.max_per_minute:
                self._sent.extend([now] * n)
                return

            wait = 60.0 - (now - self._sent[0]) + 0.5
            if wait > 0:
                print(f"    rate limit: waiting {wait:.0f}s "
                      f"({len(self._sent)}/{self.max_per_minute} used this minute)")
                time.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """After a 429, treat the whole window as spent for `seconds`."""
        now = time.monotonic()
        self._sent.clear()
        self._sent.extend([now + max(0.0, seconds - 60.0)] * self.max_per_minute)


_RETRY_PATTERNS = (
    re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)s"),
    re.compile(r"retry_delay\s*{\s*seconds:\s*([0-9]+)"),
    re.compile(r"seconds:\s*([0-9]+)"),
)


def server_retry_delay(exc: Exception) -> float | None:
    """
    Extract the wait the server asked for.

    Google's 429 says exactly how long to wait ("Please retry in 45.2s").
    Ignoring that and using your own shorter backoff guarantees the retry fails
    — which is precisely what happened before this was added.
    """
    text = str(exc)
    for pattern in _RETRY_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def is_quota_error(exc: Exception) -> bool:
    t = str(exc).lower()
    return "429" in t or "resource_exhausted" in t or "quota" in t


class ThrottledEmbeddings(Embeddings):
    """
    Rate-limited wrapper around an API embedding model.

    - Limits TEXTS per minute, not calls (see RateLimiter).
    - Honours the server's own retry_delay on 429 instead of guessing.
    - Retries generously, because a quota wait is not a failure — it is the
      normal cost of running a few thousand embeddings on a free tier.

    Query embedding is deliberately not throttled: it is a single call on the
    interactive path, and latency there is user-visible.
    """

    def __init__(
        self,
        inner: Embeddings,
        batch_size: int = config.GEMINI_EMBED_BATCH,
        rpm: int = config.GEMINI_EMBED_RPM,
        max_attempts: int = config.GEMINI_EMBED_MAX_ATTEMPTS,
        limiter: RateLimiter | None = None,
    ):
        self.inner = inner
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.limiter = limiter or RateLimiter(rpm)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from tqdm.auto import tqdm

        out: list[list[float]] = []
        starts = list(range(0, len(texts), self.batch_size))

        for start in tqdm(starts, desc="  embedding", leave=False, unit="batch"):
            batch = texts[start : start + self.batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.limiter.acquire(len(batch))
            try:
                return self.inner.embed_documents(batch)
            except Exception as exc:
                last_error = exc

                if is_quota_error(exc):
                    wait = server_retry_delay(exc) or 60.0
                    wait = min(max(wait + 2.0, 5.0), 180.0)
                    self.limiter.penalise(wait)
                    print(f"    quota hit — waiting {wait:.0f}s as the server asked "
                          f"(attempt {attempt}/{self.max_attempts})")
                else:
                    wait = min(2.0 * (2 ** (attempt - 1)), 60.0)
                    print(f"    {type(exc).__name__} — retrying in {wait:.0f}s "
                          f"(attempt {attempt}/{self.max_attempts})")

                if attempt < self.max_attempts:
                    time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {self.max_attempts} attempts: {last_error}"
        )

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------

def _bge(model_name: str = config.OSS_EMBED_MODEL, device: str | None = None) -> Embeddings:
    """
    BGE models expect an instruction prefix on QUERIES but not on documents.
    Skipping this costs a few points of retrieval quality — it's a classic
    "why is my open-source model underperforming" gotcha.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )


_SHARED_GEMINI_LIMITER: RateLimiter | None = None


def _gemini_limiter() -> RateLimiter:
    global _SHARED_GEMINI_LIMITER
    if _SHARED_GEMINI_LIMITER is None:
        _SHARED_GEMINI_LIMITER = RateLimiter(config.GEMINI_EMBED_RPM)
    return _SHARED_GEMINI_LIMITER


def _gemini(dim: int | None = None) -> Embeddings:
    """
    gemini-embedding-001 with task types.

    task_type matters: RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY produce vectors
    optimised for asymmetric search (short question vs long passage). Using
    SEMANTIC_SIMILARITY for both sides is a common and costly mistake.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.get_google_api_key()

    kwargs = {"model": config.GEMINI_EMBED_MODEL, "task_type": "RETRIEVAL_DOCUMENT"}
    if dim:
        kwargs["output_dimensionality"] = dim

    inner = GoogleGenerativeAIEmbeddings(**kwargs)
    # Both Gemini arms are the same underlying model and therefore share one
    # quota bucket — they must share one limiter, or building the second arm
    # immediately trips the limit the first one was carefully respecting.
    return ThrottledEmbeddings(inner, limiter=_gemini_limiter())


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

EMBEDDING_SPECS = {
    "bge": {
        "label": "BAAI/bge-base-en-v1.5",
        "kind": "open-source",
        "dim": config.OSS_EMBED_DIM,
        "collection": "papers_bge",
    },
    "gemini": {
        "label": "gemini-embedding-001 (3072d)",
        "kind": "commercial",
        "dim": config.GEMINI_EMBED_DIM,
        "collection": "papers_gemini",
    },
    "gemini_768": {
        "label": "gemini-embedding-001 (768d, truncated)",
        "kind": "commercial",
        "dim": config.GEMINI_EMBED_DIM_TRUNCATED,
        "collection": "papers_gemini_768",
    },
}


def get_embedder(name: str, cached: bool = True) -> Embeddings:
    """
    Build an embedder by key: "bge" | "gemini" | "gemini_768".

    cached=True wraps it in a disk-backed cache keyed by text hash, so
    re-running the notebook does not re-embed or re-spend API quota.
    """
    if name not in EMBEDDING_SPECS:
        raise ValueError(f"Unknown embedder '{name}'. Options: {list(EMBEDDING_SPECS)}")

    if name == "bge":
        base = _bge()
    elif name == "gemini":
        base = _gemini(dim=None)
    else:
        base = _gemini(dim=config.GEMINI_EMBED_DIM_TRUNCATED)

    if not cached:
        return base

    config.ensure_dirs()
    store = LocalFileStore(str(config.CACHE_DIR() / f"emb_{name}"))
    return CacheBackedEmbeddings.from_bytes_store(
        base,
        store,
        namespace=name,
        query_embedding_cache=False,   # queries are cheap and vary constantly
    )


def probe(name: str) -> dict:
    """Sanity-check an embedder: does it load, what dimension does it emit, how fast?"""
    emb = get_embedder(name, cached=False)
    t0 = time.perf_counter()
    vec = emb.embed_query("What is multi-head attention?")
    elapsed = time.perf_counter() - t0
    return {
        "embedder": name,
        "label": EMBEDDING_SPECS[name]["label"],
        "kind": EMBEDDING_SPECS[name]["kind"],
        "reported_dim": EMBEDDING_SPECS[name]["dim"],
        "actual_dim": len(vec),
        "query_latency_s": round(elapsed, 3),
    }
