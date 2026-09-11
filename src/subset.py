"""
Evaluation pooling — running Experiment 1 under a hard API quota.

THE CONSTRAINT
--------------
The Gemini free tier allows 1,000 embed_content requests per DAY per model.
Both Gemini arms (3072d and 768d) are the same model, so they share that bucket.
Embedding the full 2,176-chunk corpus twice needs ~4,350 requests — five days of
quota. Not viable inside a 30-day capstone that also needs the quota for
generation and RAGAS.

THE FIX — POOLED RELEVANCE
--------------------------
Evaluate all arms on a shared SUBSET of the corpus instead of the whole thing.
This is not a shortcut invented for convenience: it is how TREC and most IR
benchmarks have always worked. You cannot judge every document against every
query, so you pool candidate documents from one or more baseline retrievers,
judge the pool, and evaluate all systems against it.

The pool here is built from two independent sources per question:

  1. Dense retrieval (bge)  — semantic candidates
  2. BM25                   — lexical candidates, EMBEDDING-INDEPENDENT

Including BM25 matters. Pooling from a dense retriever alone would stack the
deck in favour of the embedding model that built the pool. BM25 knows nothing
about any embedding model, so roughly half the pool is chosen by a process that
cannot favour bge or Gemini.

RESIDUAL BIAS — state this in the notebook, do not hide it
----------------------------------------------------------
Half the pool still comes from bge. If bge wins Experiment 1 on this pool, that
result is confounded and should be reported as inconclusive. If GEMINI wins on a
pool half-built by its competitor, the result is strong — it won on unfavourable
ground. Either way, say which case you are in.
"""

from __future__ import annotations

import random

from langchain_core.documents import Document

from . import config


def build_eval_pool(
    docs: list[Document],
    questions: list[dict],
    dense_store,
    per_query_dense: int = 15,
    per_query_bm25: int = 15,
    target_size: int = 450,
    seed: int = 42,
) -> tuple[list[Document], dict]:
    """
    Build the shared evaluation subset.

    Returns (pool_docs, stats). `target_size` should be chosen so that
    target_size * 2 stays under the daily quota — 450 leaves headroom under
    1,000 for the two Gemini arms.
    """
    from langchain_community.retrievers import BM25Retriever

    by_id = {d.metadata["chunk_id"]: d for d in docs}
    asked = [q for q in questions if q.get("answerable", True)]

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = per_query_bm25

    dense_ids: set[str] = set()
    bm25_ids: set[str] = set()

    for q in asked:
        question = q["question"]
        for d in dense_store.similarity_search(question, k=per_query_dense):
            cid = d.metadata.get("chunk_id")
            if cid in by_id:
                dense_ids.add(cid)
        for d in bm25.invoke(question):
            cid = d.metadata.get("chunk_id")
            if cid in by_id:
                bm25_ids.add(cid)

    pooled = dense_ids | bm25_ids

    # Distractor fill: stratified by paper so every paper is represented and the
    # pool is not just "chunks some retriever already liked". Without these, a
    # retriever cannot be penalised for ranking an irrelevant chunk highly,
    # because no irrelevant chunks would exist in the index.
    rng = random.Random(seed)
    by_paper: dict[str, list[str]] = {}
    for d in docs:
        cid = d.metadata["chunk_id"]
        if cid not in pooled:
            by_paper.setdefault(d.metadata["paper_id"], []).append(cid)

    need = max(0, target_size - len(pooled))
    papers = sorted(by_paper)
    per_paper = need // max(1, len(papers)) + 1

    fill: list[str] = []
    for pid in papers:
        candidates = by_paper[pid]
        rng.shuffle(candidates)
        fill.extend(candidates[:per_paper])
    rng.shuffle(fill)
    pooled |= set(fill[:need])

    pool_docs = [by_id[cid] for cid in sorted(pooled)]

    papers_covered = {d.metadata["paper_id"] for d in pool_docs}
    stats = {
        "pool_size": len(pool_docs),
        "from_dense": len(dense_ids),
        "from_bm25": len(bm25_ids),
        "overlap_dense_bm25": len(dense_ids & bm25_ids),
        "bm25_only": len(bm25_ids - dense_ids),
        "distractors_added": min(need, len(fill)),
        "papers_covered": len(papers_covered),
        "papers_total": len({d.metadata["paper_id"] for d in docs}),
        "corpus_size": len(docs),
        "fraction_of_corpus": round(len(pool_docs) / max(1, len(docs)), 3),
    }
    return pool_docs, stats


def coverage_report(pool_docs: list[Document], questions: list[dict]) -> "object":
    """
    Per-question check that the pool can actually answer it.

    A question whose labelled paper contributes no chunk to the pool is
    unanswerable by construction and would drag every arm's score down equally —
    noise, not signal. Drop those from Experiment 1 and say how many you dropped.
    """
    import pandas as pd

    papers_in_pool = {d.metadata["paper_id"] for d in pool_docs}

    rows = []
    for q in questions:
        if not q.get("answerable", True):
            continue
        wanted = set(q["answer_paper_ids"])
        present = wanted & papers_in_pool
        rows.append(
            {
                "id": q.get("id", ""),
                "type": q.get("type", ""),
                "question": q["question"][:58],
                "papers_needed": len(wanted),
                "papers_in_pool": len(present),
                "usable": len(present) > 0,
            }
        )
    return pd.DataFrame(rows)
