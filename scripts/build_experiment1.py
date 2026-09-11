"""
Build the Experiment 1 subset indexes, within the Gemini free-tier daily quota.

WHY THIS EXISTS
---------------
The free tier allows 1,000 embed_content requests per DAY per model, and both
Gemini arms are the same model sharing one bucket. Embedding all 2,176 chunks
twice needs ~4,350 requests — five days. So Experiment 1 runs on a pooled
subset instead (see src/subset.py for the methodology and its limitations).

The FULL corpus bge index is unaffected and remains the production system.

USAGE
    python scripts/build_experiment1.py --plan          # show the quota maths, build nothing
    python scripts/build_experiment1.py                 # build the subset (~900 embeddings)
    python scripts/build_experiment1.py --size 350      # smaller pool if quota is tight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd   # noqa: E402

from src import config                                       # noqa: E402
from src.chunking import chunk_recursive                     # noqa: E402
from src.download import download_corpus                     # noqa: E402
from src.indexing import build_index, index_summary, load_index, quota_report  # noqa: E402
from src.parse import parse_corpus                           # noqa: E402
from src.evaluate import load_questions                      # noqa: E402
from src.subset import build_eval_pool, coverage_report      # noqa: E402

SUFFIX = "_sub"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=450,
                    help="target pool size (x2 arms must stay under the daily quota)")
    ap.add_argument("--plan", action="store_true",
                    help="print the quota arithmetic and exit without building")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()

    print("\n[1/5] Loading corpus (all cached)")
    papers = download_corpus()
    pages = parse_corpus(papers, parser="pymupdf", use_cache=True)
    docs = chunk_recursive(pages)
    questions = load_questions()
    print(f"    {len(docs)} chunks, {len(questions)} eval questions")

    print("\n[2/5] Quota arithmetic")
    full = quota_report(len(docs), n_api_arms=2)
    sub = quota_report(args.size, n_api_arms=2)
    print(f"    FULL corpus : {full['embeddings_needed']} embeddings "
          f"= {full['days_required']} days of free-tier quota")
    print(f"    SUBSET({args.size}): {sub['embeddings_needed']} embeddings "
          f"= {sub['days_required']} days "
          f"({'fits in one day' if sub['fits_in_one_day'] else 'DOES NOT FIT'})")
    print(f"    max pool size for a single day: {full['max_chunks_for_one_day']}")

    if not sub["fits_in_one_day"]:
        print(f"\n    Reduce --size to {full['max_chunks_for_one_day']} or below.")
        return 1

    if args.plan:
        print("\n--plan given: nothing built.")
        return 0

    print("\n[3/5] Building the pooled evaluation subset")
    bge_full = load_index("bge")
    if bge_full._collection.count() == 0:
        print("    The full bge index is empty. Run `python scripts/build_index.py` first.")
        return 1

    pool, stats = build_eval_pool(docs, questions, bge_full, target_size=args.size)
    print(json.dumps(stats, indent=4))

    cov = coverage_report(pool, questions)
    usable = int(cov["usable"].sum())
    print(f"\n    questions answerable from the pool: {usable}/{len(cov)}")
    if usable < len(cov):
        print("    not covered:")
        for _, r in cov[~cov["usable"]].iterrows():
            print(f"      {r['id']}  {r['question']}")

    config.RESULTS_DIR().mkdir(parents=True, exist_ok=True)
    cov.to_csv(config.RESULTS_DIR() / "experiment1_pool_coverage.csv", index=False)
    pd.DataFrame([stats]).to_csv(
        config.RESULTS_DIR() / "experiment1_pool_stats.csv", index=False
    )

    # Persist the pool ids so the notebook uses exactly the same subset.
    pool_ids = [d.metadata["chunk_id"] for d in pool]
    (config.CACHE_DIR() / "experiment1_pool_ids.json").write_text(
        json.dumps(pool_ids, indent=1), encoding="utf-8"
    )
    print(f"    pool ids saved -> data/cache/experiment1_pool_ids.json")

    print("\n[4/5] Indexing the subset (bge first — free, no quota)")
    build_index(pool, "bge", rebuild=args.rebuild, suffix=SUFFIX)

    print("\n[5/5] Indexing the subset (Gemini arms — uses daily quota)")
    for name in ("gemini", "gemini_768"):
        print(f"\n  --- {name} ---")
        try:
            build_index(pool, name, rebuild=args.rebuild, suffix=SUFFIX)
        except Exception as exc:
            print(f"\n  {name} stopped: {exc}")
            print("  Daily quota is likely spent. Re-run this same command tomorrow —")
            print("  completed embeddings are cached and it resumes where it stopped.")
            return 1

    print("\nSubset index summary")
    for name in ("bge", "gemini", "gemini_768"):
        print("   ", index_summary(name, SUFFIX))

    print("\nDone. Experiment 1 in the notebook should now load the _sub collections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
