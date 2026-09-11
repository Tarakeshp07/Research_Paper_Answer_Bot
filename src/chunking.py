"""
Step 2 — Chunking strategies.

Two strategies are implemented so the notebook can compare them:

  1. Recursive character splitting  — fast, predictable, the sane baseline
  2. Semantic chunking             — splits where embedding similarity drops,
                                     i.e. at topic shifts rather than at a
                                     character count

Both produce LangChain Documents whose metadata carries paper title, arXiv id,
page number and section. Metadata propagation is the whole point — a chunk that
loses its page number is worthless for citation.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .parse import PageText


def _metadata_for(page: PageText, chunk_index: int) -> dict:
    """
    Chroma metadata values must be str/int/float/bool — no lists, no None.
    Keep this function as the single place that decides chunk metadata shape.
    """
    return {
        "chunk_id": f"{page.arxiv_id}_p{page.page_number:03d}_c{chunk_index:02d}",
        "paper_id": page.arxiv_id,
        "paper_title": page.paper_title,
        "authors": page.authors or "",
        "year": int(page.year or 0),
        "page_number": int(page.page_number),
        "section": page.primary_section or "",
        "chunk_index": int(chunk_index),
    }


# --------------------------------------------------------------------------
# Strategy 1 — recursive character splitting (baseline)
# --------------------------------------------------------------------------

def chunk_recursive(
    pages: list[PageText],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    min_chars: int = config.MIN_CHUNK_CHARS,
) -> list[Document]:
    """
    Split page by page. Chunking WITHIN a page (never across pages) is what
    guarantees a chunk has exactly one page number — the alternative, splitting
    a concatenated document, produces chunks that straddle a page boundary and
    can only be cited ambiguously.

    The cost is some lost context at page breaks. That trade is worth it here
    because the rubric demands unambiguous page citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            docs.append(Document(page_content=piece, metadata=_metadata_for(page, i)))
    return docs


# --------------------------------------------------------------------------
# Strategy 2 — semantic chunking
# --------------------------------------------------------------------------

def chunk_semantic(
    pages: list[PageText],
    embeddings,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    min_chars: int = config.MIN_CHUNK_CHARS,
    max_pages: int | None = None,
) -> list[Document]:
    """
    Semantic chunking via langchain_experimental.SemanticChunker.

    WARNING: this embeds every sentence to find split points, so it is far more
    expensive than recursive splitting. Pass max_pages to run it on a subset for
    the comparison rather than the whole corpus — the notebook does exactly that.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    subset = pages[:max_pages] if max_pages else pages
    docs: list[Document] = []
    for page in subset:
        try:
            pieces = chunker.split_text(page.text)
        except Exception:
            continue
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            docs.append(Document(page_content=piece, metadata=_metadata_for(page, i)))
    return docs


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

REQUIRED_KEYS = ("chunk_id", "paper_id", "paper_title", "page_number")


def assert_chunk_metadata(docs: list[Document]) -> None:
    """Second hard gate: metadata survived chunking."""
    if not docs:
        raise AssertionError("No chunks produced.")

    for d in docs:
        for key in REQUIRED_KEYS:
            if d.metadata.get(key) in (None, ""):
                raise AssertionError(
                    f"Chunk missing '{key}': {d.metadata.get('chunk_id', '<no id>')}"
                )
        if not isinstance(d.metadata["page_number"], int) or d.metadata["page_number"] < 1:
            raise AssertionError(f"Bad page_number on {d.metadata['chunk_id']}")

    ids = [d.metadata["chunk_id"] for d in docs]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        raise AssertionError(f"{dupes} duplicate chunk_id(s) — ids must be unique for Chroma.")

    print(f"Chunk metadata check PASSED — {len(docs)} chunks, all with page numbers, ids unique")


def chunk_stats(docs: list[Document]):
    """Return a DataFrame of per-chunk stats for the EDA section."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "chunk_id": d.metadata["chunk_id"],
                "paper_id": d.metadata["paper_id"],
                "paper_title": d.metadata["paper_title"],
                "page_number": d.metadata["page_number"],
                "section": d.metadata.get("section", ""),
                "n_chars": len(d.page_content),
                "n_words": len(d.page_content.split()),
            }
            for d in docs
        ]
    )
