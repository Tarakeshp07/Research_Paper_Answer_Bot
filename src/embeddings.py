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

import time
from typing import Iterable

from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.storage import LocalFileStore

from . import config


# --------------------------------------------------------------------------
# Rate-limited wrapper for the Gemini free tier
# --------------------------------------------------------------------------

class ThrottledEmbeddings(Embeddings):
    """
    Wraps an Embeddings object, batching document calls and sleeping between
    batches so the Gemini free tier (~15 RPM) doesn't return 429s mid-index.

    Query embedding is not throttled — it's a single call and latency matters.
    """

    def __init__(
        self,
        inner: Embeddings,
        batch_size: int = config.GEMINI_EMBED_BATCH,
        sleep_s: float = config.GEMINI_EMBED_SLEEP,
    ):
        self.inner = inner
        self.batch_size = batch_size
        self.sleep_s = sleep_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from tqdm.auto import tqdm

        out: list[list[float]] = []
        batches = range(0, len(texts), self.batch_size)
        for start in tqdm(batches, desc="Embedding batches", leave=False):
            batch = texts[start : start + self.batch_size]
            for attempt in range(5):
                try:
                    out.extend(self.inner.embed_documents(batch))
                    break
                except Exception as exc:
                    wait = self.sleep_s * (2 ** attempt)
                    if attempt == 4:
                        raise
                    print(f"  embed retry {attempt + 1}/4 after {wait:.0f}s ({type(exc).__name__})")
                    time.sleep(wait)
            time.sleep(self.sleep_s)
        return out

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
    return ThrottledEmbeddings(inner)


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
