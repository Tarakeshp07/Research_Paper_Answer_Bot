"""
Step 1b — Document parsing with page provenance.

THE critical invariant of this whole project: every unit of text we extract
carries the page number it came from. The rubric asks for page-level citations
in four separate places, and page numbers cannot be recovered after the fact.

Primary parser : pymupdf4llm  (fast, markdown output, reliable page boundaries)
Comparison     : docling      (slower, better multi-column structure) — optional

Both are exposed so Section 2 of the notebook can put them side by side on a
two-column page and document the difference, which is exactly the kind of
"explore approaches, then justify your choice" evidence the brief rewards.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from . import config

# Headings that mark the end of the substantive paper. Everything after these
# is bibliography — high token count, near-zero answer value, and it pollutes
# BM25 with author surnames. We cut it by default.
_END_SECTIONS = re.compile(
    r"^\s*#{0,6}\s*(references|bibliography|acknowledg(e)?ments)\b",
    re.IGNORECASE | re.MULTILINE,
)

_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class PageText:
    """One page of one paper, with everything needed to cite it."""
    arxiv_id: str
    paper_title: str
    authors: str
    year: int
    page_number: int          # 1-indexed, matches what a human sees in a PDF reader
    text: str
    sections: list[str] = field(default_factory=list)

    @property
    def primary_section(self) -> str:
        return self.sections[0] if self.sections else ""


# --------------------------------------------------------------------------
# Parser A — pymupdf4llm (default)
# --------------------------------------------------------------------------

def parse_pymupdf(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Returns [(page_number, markdown_text), ...], 1-indexed.

    pymupdf4llm returns one dict per page in document order when
    page_chunks=True, which is what gives us dependable page provenance.
    """
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)

    out: list[tuple[int, str]] = []
    for idx, page in enumerate(pages):
        meta = page.get("metadata", {}) or {}
        # Prefer the parser's own page number; fall back to positional index.
        page_no = meta.get("page") or (idx + 1)
        out.append((int(page_no), page.get("text", "") or ""))
    return out


# --------------------------------------------------------------------------
# Parser B — docling (optional comparison arm)
# --------------------------------------------------------------------------

def parse_docling(pdf_path: str | Path, max_pages: int | None = None) -> list[tuple[int, str]]:
    """
    Structure-aware parse via Docling. Returns [(page_number, text), ...].

    Requires `pip install docling` (~1-2GB of model weights on first run).
    Raises ImportError if unavailable so the caller can skip this arm.

    max_pages: convert only the first N pages. Docling runs a layout model on
    every page and is 5-30x slower than pymupdf4llm, so a full paper takes
    minutes on CPU. The Section 2 comparison only needs to look at ONE page —
    pass max_pages=4 there and keep the cell responsive. Leave it None only if
    you actually intend to parse the whole document.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    kwargs = {}
    if max_pages:
        kwargs["page_range"] = (1, max_pages)   # 1-based, inclusive

    result = converter.convert(str(pdf_path), **kwargs)
    doc = result.document

    by_page: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        # Provenance carries the source page for each text item.
        prov = getattr(item, "prov", None) or []
        page_no = getattr(prov[0], "page_no", None) if prov else None
        if page_no is None:
            continue
        by_page.setdefault(int(page_no), []).append(text.strip())

    return [(p, "\n\n".join(by_page[p])) for p in sorted(by_page)]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _truncate_at_references(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop everything from the first References/Bibliography heading onward."""
    cut_page: int | None = None
    cut_offset = 0

    for page_no, text in pages:
        m = _END_SECTIONS.search(text)
        if m:
            # Only treat it as the real bibliography if it appears past the
            # halfway mark — some papers cite "references" in the body.
            if page_no >= max(2, len(pages) // 2):
                cut_page, cut_offset = page_no, m.start()
                break

    if cut_page is None:
        return pages

    kept: list[tuple[int, str]] = []
    for page_no, text in pages:
        if page_no < cut_page:
            kept.append((page_no, text))
        elif page_no == cut_page:
            head = text[:cut_offset].strip()
            if head:
                kept.append((page_no, head))
            break
    return kept


def _sections_on_page(markdown: str) -> list[str]:
    """Extract markdown headings so each chunk can name its section."""
    return [f"{m.group(2).strip()}" for m in _MD_HEADER.finditer(markdown)]


def parse_paper(
    meta,                       # PaperMeta from download.py (or any obj with the fields)
    parser: str = "pymupdf",
    drop_references: bool = True,
) -> list[PageText]:
    """Parse a single paper into page-tagged text."""
    fn = {"pymupdf": parse_pymupdf, "docling": parse_docling}[parser]
    pages = fn(meta.pdf_path)

    if drop_references:
        pages = _truncate_at_references(pages)

    out: list[PageText] = []
    for page_no, text in pages:
        if not text or not text.strip():
            continue
        out.append(
            PageText(
                arxiv_id=meta.arxiv_id,
                paper_title=meta.title,
                authors=meta.authors,
                year=meta.year,
                page_number=page_no,
                text=text,
                sections=_sections_on_page(text),
            )
        )
    return out


def parse_corpus(
    papers: list,
    parser: str = "pymupdf",
    drop_references: bool = True,
    use_cache: bool = True,
) -> list[PageText]:
    """
    Parse every paper, with a disk cache.

    Parsing is deterministic and slow-ish; the notebook gets re-run many times.
    Cache the result so a full re-run costs seconds, not minutes.
    """
    config.ensure_dirs()
    cache_file = config.CACHE_DIR() / f"parsed_{parser}_{'noref' if drop_references else 'full'}.pkl"

    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            pages = pickle.load(f)
        print(f"Loaded {len(pages)} pages from cache ({cache_file.name})")
        return pages

    all_pages: list[PageText] = []
    for meta in tqdm(papers, desc=f"Parsing ({parser})"):
        try:
            all_pages.extend(parse_paper(meta, parser=parser, drop_references=drop_references))
        except Exception as exc:
            print(f"  !! {meta.arxiv_id}: parse failed — {exc}")

    with open(cache_file, "wb") as f:
        pickle.dump(all_pages, f)

    print(f"Parsed {len(all_pages)} pages from {len(papers)} papers -> cached")
    return all_pages


def assert_page_numbers(pages: list[PageText]) -> None:
    """
    Hard gate. Run this immediately after parsing, before anything else is built.

    If this raises, stop and fix parsing. Every downstream rubric item that
    mentions citations depends on it passing.
    """
    missing = [p for p in pages if not p.page_number or p.page_number < 1]
    if missing:
        raise AssertionError(
            f"{len(missing)} page(s) have no valid page_number. "
            f"First offender: {missing[0].arxiv_id}"
        )

    per_paper: dict[str, int] = {}
    for p in pages:
        per_paper[p.arxiv_id] = per_paper.get(p.arxiv_id, 0) + 1

    print(f"Page-number check PASSED — {len(pages)} pages across {len(per_paper)} papers")
    for aid, n in sorted(per_paper.items()):
        print(f"    {aid}: {n} pages")
