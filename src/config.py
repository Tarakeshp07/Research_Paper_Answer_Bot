"""
Central configuration for the Research Paper Answer Bot.

Everything tunable lives here so the notebook never hardcodes a path,
a model id, or a magic number. Import this module, don't copy values out of it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

load_dotenv()


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = _in_colab()


def get_google_api_key() -> str:
    """
    Resolve the Gemini API key.

    Order of preference:
      1. Colab secrets (userdata) — the safe option on Colab
      2. GOOGLE_API_KEY environment variable / .env file
    """
    if IN_COLAB:
        try:
            from google.colab import userdata
            key = userdata.get("GOOGLE_API_KEY")
            if key:
                os.environ["GOOGLE_API_KEY"] = key
                return key
        except Exception:
            pass

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found.\n"
            "  Local : copy .env.example to .env and add your key\n"
            "  Colab : add GOOGLE_API_KEY to the Secrets panel (key icon, left sidebar)"
        )
    return key


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# PROJECT_ROOT is the directory containing src/, app/, notebooks/, data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# On Colab, override with the Drive path so a runtime disconnect costs nothing.
# e.g. config.set_project_root("/content/drive/MyDrive/capstone_rag")
_ROOT_OVERRIDE: Path | None = None


def set_project_root(path: str | Path) -> None:
    """Point all derived paths at a different root (used on Colab + Drive)."""
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = Path(path).resolve()
    _ROOT_OVERRIDE.mkdir(parents=True, exist_ok=True)
    for d in (PAPERS_DIR(), CACHE_DIR(), EVAL_DIR(), RESULTS_DIR(), CHROMA_DIR()):
        d.mkdir(parents=True, exist_ok=True)


def root() -> Path:
    return _ROOT_OVERRIDE or PROJECT_ROOT


def DATA_DIR() -> Path:      return root() / "data"
def PAPERS_DIR() -> Path:    return DATA_DIR() / "papers"
def CACHE_DIR() -> Path:     return DATA_DIR() / "cache"
def EVAL_DIR() -> Path:      return DATA_DIR() / "eval"
def CHROMA_DIR() -> Path:    return DATA_DIR() / "chroma"
def RESULTS_DIR() -> Path:   return root() / "results"


def ensure_dirs() -> None:
    for d in (PAPERS_DIR(), CACHE_DIR(), EVAL_DIR(), CHROMA_DIR(), RESULTS_DIR()):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Open-source embedding model (Experiment 1, arm A)
OSS_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
OSS_EMBED_DIM = 768

# Commercial embedding model (Experiment 1, arm B)
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_EMBED_DIM = 3072
# Third arm: Gemini truncated to the OSS model's dimensionality, so the
# comparison isolates model quality from vector width.
GEMINI_EMBED_DIM_TRUNCATED = 768

# Generation model
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0          # grounded QA — we do not want creativity

# Cross-encoder reranker (CPU-friendly; swap for BAAI/bge-reranker-v2-m3 on GPU)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

CHUNK_SIZE = 1000              # characters
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 100          # drop fragments shorter than this (headers, page numbers)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

TOP_K = 3                      # passages shown to the user (rubric: top-3)
FETCH_K = 20                   # candidates pulled before MMR / reranking
MMR_LAMBDA = 0.5
HYBRID_WEIGHTS = (0.5, 0.5)    # (bm25, dense) for the ensemble retriever

# --------------------------------------------------------------------------
# Gemini free-tier throttling
# --------------------------------------------------------------------------
# Free tier is roughly 15 RPM / 1500 RPD for Flash. We batch embeddings and
# sleep between batches so a full index build doesn't trip the limiter.
GEMINI_EMBED_BATCH = 64
GEMINI_EMBED_SLEEP = 1.0       # seconds between batches

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
# 14 seminal GenAI papers. Deliberately includes pairs that disagree
# (Kaplan vs Chinchilla, InstructGPT vs DPO) so the eval set can contain
# genuine cross-paper questions rather than only lookups.
#
# NOTE: these arXiv ids are verified at download time — download.py fetches
# each paper's real title from the arXiv API and flags any mismatch loudly.

CORPUS = [
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need",                                   "short": "Transformer"},
    {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers",       "short": "BERT"},
    {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners",                       "short": "GPT-3"},
    {"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",  "short": "RAG"},
    {"arxiv_id": "2004.04906", "title": "Dense Passage Retrieval for Open-Domain Question Answering",  "short": "DPR"},
    {"arxiv_id": "2001.08361", "title": "Scaling Laws for Neural Language Models",                     "short": "Kaplan Scaling"},
    {"arxiv_id": "2203.15556", "title": "Training Compute-Optimal Large Language Models",              "short": "Chinchilla"},
    {"arxiv_id": "2201.11903", "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",        "short": "CoT"},
    {"arxiv_id": "2203.02155", "title": "Training Language Models to Follow Instructions",             "short": "InstructGPT"},
    {"arxiv_id": "2305.18290", "title": "Direct Preference Optimization",                              "short": "DPO"},
    {"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models",          "short": "LoRA"},
    {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models",  "short": "ReAct"},
    {"arxiv_id": "2310.11511", "title": "Self-RAG: Learning to Retrieve, Generate and Critique",       "short": "Self-RAG"},
    {"arxiv_id": "2307.03172", "title": "Lost in the Middle: How Language Models Use Long Contexts",   "short": "Lost in Middle"},
]

CORPUS_IDS = [p["arxiv_id"] for p in CORPUS]
