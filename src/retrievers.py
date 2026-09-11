"""
Step 5 — Retrieval strategies.

Four strategies, one factory, so Experiment 2 can loop over them uniformly:

  dense   Cosine similarity over the vector index. The baseline.
  mmr     Maximal Marginal Relevance — trades some relevance for diversity,
          which matters when the top-5 are five near-copies of one paragraph.
  hybrid  BM25 (lexical) + dense (semantic), merged by Reciprocal Rank Fusion.
          Catches exact-term queries ("LoRA rank r", "BLEU") that embeddings
          blur, while keeping semantic recall.
  rerank  hybrid -> cross-encoder rerank. A bi-encoder scores query and passage
          SEPARATELY; a cross-encoder scores them JOINTLY, which is far more
          accurate but too slow to run over the whole corpus. So: retrieve 20
          cheaply, rerank to 3 expensively. This is usually the winner.
"""

from __future__ import annotations

import time
from typing import Callable

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import config

STRATEGIES = ("dense", "mmr", "hybrid", "rerank")

STRATEGY_LABELS = {
    "dense": "Dense cosine",
    "mmr": f"MMR (fetch_k={config.FETCH_K}, lambda={config.MMR_LAMBDA})",
    "hybrid": "Hybrid BM25 + dense (RRF)",
    "rerank": "Hybrid + cross-encoder rerank",
}

_RERANKER_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------
# Individual strategies
# --------------------------------------------------------------------------

def dense_retriever(store, k: int = config.TOP_K) -> BaseRetriever:
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def mmr_retriever(store, k: int = config.TOP_K) -> BaseRetriever:
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": config.FETCH_K,
            "lambda_mult": config.MMR_LAMBDA,
        },
    )


def bm25_retriever(docs: list[Document], k: int = config.TOP_K) -> BM25Retriever:
    """
    Lexical retrieval. BM25 needs the raw corpus in memory — it builds its own
    inverted index and knows nothing about Chroma.
    """
    r = BM25Retriever.from_documents(docs)
    r.k = k
    return r


def hybrid_retriever(store, docs: list[Document], k: int = config.TOP_K) -> BaseRetriever:
    """
    EnsembleRetriever merges ranked lists with Reciprocal Rank Fusion:

        RRF(d) = sum over retrievers of  weight / (rank_of_d_in_that_list + 60)

    RRF is used instead of averaging scores because BM25 scores (unbounded) and
    cosine similarities ([0,1]) are not on a comparable scale. RRF only looks at
    RANKS, which sidesteps normalisation entirely. That is the reason to prefer
    it, and a good thing to be able to say out loud in the viva.
    """
    return EnsembleRetriever(
        retrievers=[bm25_retriever(docs, k=config.FETCH_K), dense_retriever(store, k=config.FETCH_K)],
        weights=list(config.HYBRID_WEIGHTS),
    )


def get_reranker(model_name: str = config.RERANKER_MODEL):
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = HuggingFaceCrossEncoder(model_name=model_name)
    return _RERANKER_CACHE[model_name]


def rerank_retriever(store, docs: list[Document], k: int = config.TOP_K) -> BaseRetriever:
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    base = hybrid_retriever(store, docs, k=config.FETCH_K)
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=k)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def get_retriever(
    strategy: str,
    store,
    docs: list[Document] | None = None,
    k: int = config.TOP_K,
) -> BaseRetriever:
    """
    Build any strategy by name. `docs` is required for hybrid and rerank
    (BM25 needs the corpus); dense and mmr ignore it.
    """
    if strategy == "dense":
        return dense_retriever(store, k)
    if strategy == "mmr":
        return mmr_retriever(store, k)
    if strategy in ("hybrid", "rerank"):
        if docs is None:
            raise ValueError(f"strategy '{strategy}' needs the chunk list for BM25")
        if strategy == "hybrid":
            # EnsembleRetriever has no k of its own; slice its output downstream.
            return _TopK(hybrid_retriever(store, docs, k), k)
        return rerank_retriever(store, docs, k)
    raise ValueError(f"Unknown strategy '{strategy}'. Options: {STRATEGIES}")


class _TopK(BaseRetriever):
    """Trim any retriever's output to k documents (EnsembleRetriever returns more)."""

    inner: BaseRetriever
    k: int

    def __init__(self, inner: BaseRetriever, k: int):
        super().__init__(inner=inner, k=k)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.inner.invoke(query)[: self.k]


# --------------------------------------------------------------------------
# Timing helper for the experiment table
# --------------------------------------------------------------------------

def timed_retrieve(retriever: BaseRetriever, query: str) -> tuple[list[Document], float]:
    t0 = time.perf_counter()
    docs = retriever.invoke(query)
    return docs, time.perf_counter() - t0
