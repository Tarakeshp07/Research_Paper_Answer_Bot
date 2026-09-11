"""
Step 7 — Evaluation.

Two tiers, deliberately separated:

TIER 1 — Retrieval metrics (hit rate, MRR, nDCG)
    Deterministic, free, runs in seconds. No LLM involved. Use these for ALL
    retriever and embedding tuning. Computed against hand-labelled ground truth
    in data/eval/questions.json.

TIER 2 — Generation metrics (RAGAS)
    Faithfulness, answer relevancy, context precision/recall. Needs an LLM judge,
    so it costs quota and takes minutes. Run it only on the final few configs.

Doing tuning with Tier 1 and validation with Tier 2 is the cost-aware pattern.
Running RAGAS on every experiment would exhaust the Gemini free tier by day three.

GROUND-TRUTH DESIGN
-------------------
Questions are labelled at PAPER level (`answer_paper_ids`), not chunk level.
Chunk ids depend on your chunking parameters, so chunk-level labels would be
invalidated every time you change chunk_size. Paper-level labels are stable
across every experiment, which is what makes the comparison tables comparable.

`answer_pages` is optional and starts empty. Fill it in after your first
retrieval run (there's a helper below) if you want page-level precision as an
extra column.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from tqdm.auto import tqdm

from . import config
from .retrievers import STRATEGY_LABELS, get_retriever, timed_retrieve


# --------------------------------------------------------------------------
# Eval set loading
# --------------------------------------------------------------------------

def load_questions(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else config.EVAL_DIR() / "questions.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def answerable(questions: list[dict]) -> list[dict]:
    """Only questions the corpus can actually answer are scored for retrieval."""
    return [q for q in questions if q.get("answerable", True)]


# --------------------------------------------------------------------------
# TIER 1 — retrieval metrics
# --------------------------------------------------------------------------

def _is_relevant(doc: Document, question: dict) -> bool:
    """
    A retrieved chunk is relevant if it comes from a paper labelled as
    containing the answer — and, when page labels exist, from within
    +/-1 page of a labelled page (papers split ideas across a page break).
    """
    if doc.metadata.get("paper_id") not in question["answer_paper_ids"]:
        return False

    pages = question.get("answer_pages") or []
    if not pages:
        return True

    page = doc.metadata.get("page_number")
    return any(abs(page - p) <= 1 for p in pages)


def hit_at_k(docs: list[Document], question: dict, k: int) -> int:
    return int(any(_is_relevant(d, question) for d in docs[:k]))


def reciprocal_rank(docs: list[Document], question: dict) -> float:
    for i, d in enumerate(docs, 1):
        if _is_relevant(d, question):
            return 1.0 / i
    return 0.0


def ndcg_at_k(docs: list[Document], question: dict, k: int) -> float:
    """Binary-relevance nDCG. Ideal DCG assumes all k slots could be relevant."""
    dcg = sum(
        (1.0 / math.log2(i + 1))
        for i, d in enumerate(docs[:k], 1)
        if _is_relevant(d, question)
    )
    n_rel = min(k, len(question["answer_paper_ids"]) * 2)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(
    retriever,
    questions: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5),
    label: str = "",
) -> dict:
    """Run every question through one retriever and aggregate the metrics."""
    qs = answerable(questions)
    max_k = max(k_values)

    hits = {k: [] for k in k_values}
    rrs, ndcgs, latencies = [], [], []

    for q in qs:
        retrieved, elapsed = timed_retrieve(retriever, q["question"])
        latencies.append(elapsed)
        for k in k_values:
            hits[k].append(hit_at_k(retrieved, q, k))
        rrs.append(reciprocal_rank(retrieved, q))
        ndcgs.append(ndcg_at_k(retrieved, q, max_k))

    row = {"strategy": label, "n_questions": len(qs)}
    for k in k_values:
        row[f"hit@{k}"] = round(sum(hits[k]) / len(qs), 3)
    row["MRR"] = round(sum(rrs) / len(qs), 3)
    row[f"nDCG@{max_k}"] = round(sum(ndcgs) / len(qs), 3)
    row["mean_latency_s"] = round(sum(latencies) / len(latencies), 3)
    return row


def compare_strategies(
    store,
    docs: list[Document],
    questions: list[dict],
    strategies: tuple[str, ...] = ("dense", "mmr", "hybrid", "rerank"),
    k: int = config.TOP_K,
) -> pd.DataFrame:
    """EXPERIMENT 2 — the retrieval strategy comparison table."""
    rows = []
    for s in strategies:
        print(f"  evaluating: {STRATEGY_LABELS.get(s, s)}")
        retriever = get_retriever(s, store, docs, k=max(k, 5))
        rows.append(evaluate_retriever(retriever, questions, label=STRATEGY_LABELS.get(s, s)))
    return pd.DataFrame(rows).sort_values("MRR", ascending=False).reset_index(drop=True)


def compare_embedders(
    stores: dict[str, object],
    docs: list[Document],
    questions: list[dict],
    index_times: dict[str, float] | None = None,
    strategy: str = "dense",
) -> pd.DataFrame:
    """EXPERIMENT 1 — the embedding model comparison table."""
    from .embeddings import EMBEDDING_SPECS

    rows = []
    for name, store in stores.items():
        spec = EMBEDDING_SPECS[name]
        print(f"  evaluating: {spec['label']}")
        retriever = get_retriever(strategy, store, docs, k=5)
        row = evaluate_retriever(retriever, questions, label=spec["label"])
        row.update(
            {
                "embedder": name,
                "kind": spec["kind"],
                "dim": spec["dim"],
                "index_time_s": round((index_times or {}).get(name, float("nan")), 1),
            }
        )
        rows.append(row)

    cols = ["embedder", "label", "kind", "dim", "index_time_s",
            "hit@1", "hit@3", "hit@5", "MRR", "nDCG@5", "mean_latency_s"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("MRR", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Refusal behaviour — the unanswerable questions
# --------------------------------------------------------------------------

def evaluate_refusals(chain, questions: list[dict]) -> pd.DataFrame:
    """
    Does the system correctly say "I don't know" for questions the corpus
    cannot answer? The brief names over-confident answering as a common
    failure, so having this table is a differentiator.
    """
    from .rag_chain import ask

    rows = []
    for q in [x for x in questions if not x.get("answerable", True)]:
        res = ask(chain, q["question"])
        rows.append(
            {
                "question": q["question"],
                "refused_correctly": res["refused"],
                "answer": res["answer"][:200],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# TIER 2 — RAGAS
# --------------------------------------------------------------------------

def run_ragas(
    chain,
    questions: list[dict],
    max_questions: int | None = 10,
    label: str = "config",
) -> pd.DataFrame:
    """
    EXPERIMENT 3 — end-to-end generation quality.

    Uses Gemini as both the judge LLM and the embedding model so the whole
    evaluation stays inside the free tier. Runs slowly (one judge call per
    metric per question) — keep max_questions modest.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print(
            "RAGAS is not installed, so section 8.2 is skipped.\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Install with:  pip install -r requirements-eval.txt\n"
            "  Everything else in this notebook runs without it."
        )
        empty = pd.DataFrame([{"config": label, "faithfulness": None,
                               "answer_relevancy": None, "context_precision": None,
                               "context_recall": None}])
        return empty, pd.DataFrame()

    from .embeddings import get_embedder
    from .rag_chain import ask, get_llm

    qs = answerable(questions)[:max_questions] if max_questions else answerable(questions)

    records = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}
    for q in tqdm(qs, desc=f"Generating answers [{label}]"):
        res = ask(chain, q["question"])
        records["user_input"].append(q["question"])
        records["response"].append(res["answer"])
        records["retrieved_contexts"].append([d.page_content for d in res["_docs"]])
        records["reference"].append(q.get("reference_answer", ""))
        time.sleep(0.5)   # stay inside the free-tier RPM

    dataset = Dataset.from_dict(records)

    judge = LangchainLLMWrapper(get_llm(temperature=0.0))
    judge_emb = LangchainEmbeddingsWrapper(get_embedder("gemini_768", cached=True))

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=judge_emb,
    )

    df = result.to_pandas()
    summary = {"config": label}
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if metric in df.columns:
            summary[metric] = round(float(df[metric].mean(skipna=True)), 3)
    return pd.DataFrame([summary]), df


# --------------------------------------------------------------------------
# Failure analysis
# --------------------------------------------------------------------------

def failure_cases(retriever, questions: list[dict], k: int = 3) -> pd.DataFrame:
    """
    Which questions does retrieval miss entirely? The brief asks you to
    "document failure cases — where does the system struggle?" This produces
    that table instead of you guessing.
    """
    rows = []
    for q in answerable(questions):
        docs = retriever.invoke(q["question"])
        if not hit_at_k(docs, q, k):
            rows.append(
                {
                    "question": q["question"],
                    "type": q.get("type", ""),
                    "expected_papers": ", ".join(q["answer_paper_ids"]),
                    "retrieved_papers": ", ".join(
                        dict.fromkeys(d.metadata.get("paper_id", "?") for d in docs[:k])
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, name: str) -> Path:
    config.ensure_dirs()
    path = config.RESULTS_DIR() / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved -> {path}")
    return path
