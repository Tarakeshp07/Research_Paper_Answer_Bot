"""
Step 1 — Data collection.

Downloads the corpus from arXiv and VERIFIES each paper's identity by fetching
its real title from the arXiv API and comparing it to what we expected.

Why the verification matters: an arXiv id typo silently gives you a different
paper. You would not notice until a demo question returned a confidently wrong
answer sourced from a paper you never meant to index. Fail loudly, on day one.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from . import config

ARXIV_API = "http://export.arxiv.org/api/query"
# Two URL forms — arXiv serves both, but one occasionally fails where the
# other succeeds behind proxies and campus networks.
ARXIV_PDF_URLS = [
    "https://arxiv.org/pdf/{arxiv_id}",
    "https://arxiv.org/pdf/{arxiv_id}v1",
]
ATOM = "{http://www.w3.org/2005/Atom}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; capstone-rag/1.0; academic coursework)",
    "Accept": "application/pdf,*/*",
}

# arXiv resets connections under rapid sequential fetching. Be patient rather
# than fast: this runs once and the results are cached on disk forever after.
POLITE_DELAY = 3.0          # seconds between papers
MAX_ATTEMPTS = 5            # per paper, across both URL forms


def _session() -> requests.Session:
    """
    Session with connection pooling and transport-level retries.

    ConnectionResetError (WinError 10054) is a TRANSPORT failure, not an HTTP
    status, so `status_forcelist` alone will not catch it — `connect` and `read`
    retries are what handle a reset mid-handshake.
    """
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=2.0,          # 0s, 2s, 4s, 8s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: str
    year: int
    short: str
    pdf_path: str
    expected_title: str
    title_matches: bool


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and whitespace — for fuzzy title comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _title_agrees(expected: str, actual: str) -> bool:
    """
    Loose containment check. Our CORPUS titles are often shortened
    (e.g. "BERT: Pre-training of Deep Bidirectional Transformers" vs the full
    "...Transformers for Language Understanding"), so we check whether the
    shorter normalised title is a prefix-ish substring of the longer one.
    """
    e, a = _normalise(expected), _normalise(actual)
    if not e or not a:
        return False
    short, long = (e, a) if len(e) <= len(a) else (a, e)
    if short in long:
        return True
    # Fall back to token overlap — tolerant of reordered subtitles.
    et, at = set(e.split()), set(a.split())
    overlap = len(et & at) / max(1, min(len(et), len(at)))
    return overlap >= 0.7


def fetch_metadata(arxiv_ids: list[str], session: requests.Session | None = None) -> dict[str, dict]:
    """Query the arXiv API once for all ids. Returns {arxiv_id: {title, authors, year}}."""
    session = session or _session()
    resp = session.get(
        ARXIV_API,
        params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)},
        timeout=60,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    out: dict[str, dict] = {}
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", "")
        # id looks like http://arxiv.org/abs/1706.03762v7 -> 1706.03762
        m = re.search(r"abs/([0-9]+\.[0-9]+)", raw_id)
        if not m:
            continue
        aid = m.group(1)

        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else 0

        names = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ]
        if len(names) > 2:
            authors = f"{names[0].split()[-1]} et al."
        elif names:
            authors = " & ".join(n.split()[-1] for n in names)
        else:
            authors = "Unknown"

        out[aid] = {"title": title, "authors": authors, "year": year}

    return out


def is_valid_pdf(path: Path, min_bytes: int = 10_000) -> bool:
    """A PDF is usable if it exists, is not a truncated stub, and starts with %PDF."""
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF")
    except OSError:
        return False


def download_pdf(
    arxiv_id: str,
    dest_dir: Path,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """
    Download one PDF, resiliently.

    Already-valid files are skipped, so re-running after a network failure
    resumes rather than starting over. Downloads stream to a .part file and are
    only renamed into place once complete — a killed run never leaves a
    half-written PDF that looks valid to the next run.
    """
    dest = dest_dir / f"{arxiv_id}.pdf"
    if is_valid_pdf(dest) and not overwrite:
        return dest

    session = session or _session()
    tmp = dest.with_suffix(".part")
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        url = ARXIV_PDF_URLS[(attempt - 1) % len(ARXIV_PDF_URLS)].format(arxiv_id=arxiv_id)
        try:
            with session.get(url, timeout=(20, 180), stream=True) as resp:
                resp.raise_for_status()

                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)

            if not is_valid_pdf(tmp):
                raise RuntimeError("downloaded file is not a valid PDF")

            tmp.replace(dest)
            return dest

        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                wait = min(2.0 * (2 ** (attempt - 1)), 30.0)
                print(f"      {arxiv_id}: attempt {attempt}/{MAX_ATTEMPTS} failed "
                      f"({type(exc).__name__}) — retrying in {wait:.0f}s")
                time.sleep(wait)

    raise RuntimeError(f"{arxiv_id}: all {MAX_ATTEMPTS} attempts failed — {last_error}")


def download_corpus(corpus: list[dict] | None = None, overwrite: bool = False) -> list[PaperMeta]:
    """
    Download every paper in the corpus and verify its title.

    Returns a list of PaperMeta. Any entry with title_matches=False needs
    your attention before you index anything.
    """
    corpus = corpus or config.CORPUS
    config.ensure_dirs()
    papers_dir = config.PAPERS_DIR()
    session = _session()

    ids = [p["arxiv_id"] for p in corpus]
    print(f"Fetching metadata for {len(ids)} papers from the arXiv API...")
    meta = fetch_metadata(ids, session=session)

    results: list[PaperMeta] = []
    failed: list[tuple[str, str]] = []

    for entry in tqdm(corpus, desc="Downloading PDFs"):
        aid = entry["arxiv_id"]
        info = meta.get(aid)

        if info is None:
            print(f"  !! {aid}: arXiv returned no metadata — is the id valid?")
            info = {"title": "UNKNOWN", "authors": "Unknown", "year": 0}

        dest = papers_dir / f"{aid}.pdf"
        already_had = is_valid_pdf(dest) and not overwrite

        try:
            path = download_pdf(aid, papers_dir, overwrite=overwrite, session=session)
        except Exception as exc:
            # One unreachable paper must not abort the other thirteen.
            failed.append((aid, str(exc)))
            continue

        results.append(
            PaperMeta(
                arxiv_id=aid,
                title=info["title"],          # authoritative title from arXiv
                authors=info["authors"],
                year=info["year"],
                short=entry.get("short", aid),
                pdf_path=str(path),
                expected_title=entry["title"],
                title_matches=_title_agrees(entry["title"], info["title"]),
            )
        )

        # Only pause when we actually hit the network.
        if not already_had:
            time.sleep(POLITE_DELAY)

    _report(results, failed)

    if failed:
        raise RuntimeError(
            f"{len(failed)} paper(s) could not be downloaded: "
            f"{', '.join(a for a, _ in failed)}. "
            "Re-run this script — completed downloads are kept and skipped."
        )

    return results


def _report(results: list[PaperMeta], failed: list[tuple[str, str]] | None = None) -> None:
    failed = failed or []
    print("\n" + "=" * 78)
    print("CORPUS VERIFICATION")
    print("=" * 78)
    for r in results:
        flag = "OK " if r.title_matches else "!! "
        print(f"{flag}{r.arxiv_id}  {r.title[:62]}")
        if not r.title_matches:
            print(f"      expected: {r.expected_title}")

    for aid, err in failed:
        print(f"DL {aid}  DOWNLOAD FAILED — {err[:90]}")

    bad = [r for r in results if not r.title_matches]
    print("-" * 78)
    if bad:
        print(f"{len(bad)} TITLE MISMATCH(ES) — fix the arxiv_id in config.CORPUS.")
    if failed:
        print(f"{len(failed)} DOWNLOAD FAILURE(S) — re-run to resume; finished files are skipped.")
    if not bad and not failed:
        print(f"All {len(results)} papers downloaded and verified.")
    print("=" * 78 + "\n")


def corpus_status(corpus: list[dict] | None = None) -> None:
    """
    Show which PDFs are already on disk. Useful after an interrupted run --
    tells you how much is left without touching the network.
    """
    corpus = corpus or config.CORPUS
    papers_dir = config.PAPERS_DIR()

    have, missing = [], []
    for entry in corpus:
        aid = entry["arxiv_id"]
        path = papers_dir / f"{aid}.pdf"
        if is_valid_pdf(path):
            have.append((aid, entry.get("short", ""), path.stat().st_size // 1024))
        else:
            missing.append((aid, entry.get("short", "")))

    print(f"Downloaded: {len(have)}/{len(corpus)}")
    for aid, short, kb in have:
        print(f"   OK      {aid}  {short:<18} {kb:>6} KB")
    for aid, short in missing:
        print(f"   MISSING {aid}  {short}")
    if missing:
        print("\nRun `python scripts/build_index.py` again to fetch the missing ones.")


if __name__ == "__main__":
    download_corpus()
