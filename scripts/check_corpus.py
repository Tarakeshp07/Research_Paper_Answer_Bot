"""
Show which corpus PDFs are already downloaded. Touches no network.

    python scripts/check_corpus.py

Use this after an interrupted download to see what's left.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config          # noqa: E402
from src.download import corpus_status   # noqa: E402

if __name__ == "__main__":
    config.ensure_dirs()
    print(f"Papers directory: {config.PAPERS_DIR()}\n")
    corpus_status()
