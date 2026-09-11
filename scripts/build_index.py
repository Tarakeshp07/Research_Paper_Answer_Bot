"""
One-shot pipeline: download -> parse -> chunk -> index.

Run this once before opening the notebook or launching the Streamlit app.
Everything is cached, so re-running is cheap and safe.

    python scripts/build_index.py                  # bge only (free, no API calls)
    python scripts/build_index.py --all            # bge + both Gemini arms
    python scripts/build_index.py --rebuild        # drop and rebuild indexes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                                        # noqa: E402
from src.chunking import assert_chunk_metadata, chunk_recursive, chunk_stats  # noqa: E402
from src.download import download_corpus                      # noqa: E402
from src.indexing import build_index, index_summary           # noqa: E402
from src.parse import assert_page_numbers, parse_corpus       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also build the two Gemini collections (uses API quota)")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop existing collections first")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the parse cache")
    args = ap.parse_args()

    config.ensure_dirs()

    print("\n[1/4] Downloading and verifying corpus")
    papers = download_corpus()
    if any(not p.title_matches for p in papers):
        print("Refusing to continue with unverified papers. Fix config.CORPUS first.")
        return 1

    print("\n[2/4] Parsing PDFs with page provenance")
    pages = parse_corpus(papers, parser="pymupdf", use_cache=not args.no_cache)
    assert_page_numbers(pages)

    print("\n[3/4] Chunking")
    docs = chunk_recursive(pages)
    assert_chunk_metadata(docs)
    stats = chunk_stats(docs)
    print(f"    {len(docs)} chunks | mean {stats.n_chars.mean():.0f} chars "
          f"| median {stats.n_chars.median():.0f} | max {stats.n_chars.max()}")

    print("\n[4/4] Indexing")
    targets = ["bge"] + (["gemini", "gemini_768"] if args.all else [])
    for name in targets:
        build_index(docs, name, rebuild=args.rebuild)

    print("\nIndex summary")
    for name in targets:
        print("   ", index_summary(name))

    print("\nDone. Next:")
    print("   notebooks/Research_Paper_Answer_Bot.ipynb   (the graded deliverable)")
    print("   streamlit run app/streamlit_app.py          (the demo UI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
