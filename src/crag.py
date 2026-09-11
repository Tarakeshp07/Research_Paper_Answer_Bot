"""
Stretch goal 3 — Corrective RAG (CRAG).

The failure mode CRAG fixes: a vector store ALWAYS returns its top-k, even when
nothing in the corpus is relevant. The generator then either hallucinates from
weak context or refuses, and the user learns nothing.

CRAG adds a grading step between retrieval and generation:

    retrieve -> grade each passage -> decide
                                      |
       CORRECT    (>=1 relevant)      -> generate from corpus passages only
       AMBIGUOUS  (some relevance)    -> generate from corpus + web results
       INCORRECT  (none relevant)     -> generate from web results only

Web results are always visibly labelled so the user can tell corpus-grounded
claims from web-grounded ones. That labelling matters: without it CRAG quietly
breaks the "answer only from the indexed papers" guarantee.

Uses DuckDuckGo via `ddgs` — no API key, unlike Tavily or SerpAPI.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .rag_chain import REFUSAL, format_sources, get_llm

GRADE_SYSTEM = """You grade whether a retrieved passage is relevant to a user question.

Reply with exactly one word:
  yes  - the passage contains information that helps answer the question
  no   - the passage is off-topic or contains nothing useful

Be strict. A passage that merely shares vocabulary with the question is 'no'."""

GRADE_USER = """Passage:
{passage}

Question: {question}

Relevant (yes/no):"""


CRAG_SYSTEM = f"""You are a research assistant answering questions about AI research papers.

You are given passages from two kinds of source:
  - INDEXED PAPER passages, from the curated research corpus
  - WEB passages, retrieved as a fallback because the corpus lacked coverage

RULES:
1. Answer only from the provided passages.
2. Prefer INDEXED PAPER passages. Use WEB passages only to fill genuine gaps.
3. Cite paper passages as [Paper Title, p.N] and web passages as [Web: source].
4. If the question is answered wholly or partly from the web, open your answer \
with: "Note: the indexed papers did not fully cover this; part of the answer \
comes from a web search."
5. If neither source supports an answer, reply exactly: "{REFUSAL}"
"""

CRAG_USER = """Passages:

{context}

---

Question: {question}

Answer:"""


@dataclass
class CRAGResult:
    question: str
    answer: str
    decision: str                 # CORRECT | AMBIGUOUS | INCORRECT
    n_relevant: int
    n_retrieved: int
    used_web: bool
    sources: list
    web_sources: list


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def grade_documents(docs: list[Document], question: str, llm=None) -> list[bool]:
    """One cheap LLM call per passage. Returns a relevance mask."""
    llm = llm or get_llm()
    chain = (
        ChatPromptTemplate.from_messages([("system", GRADE_SYSTEM), ("human", GRADE_USER)])
        | llm
        | StrOutputParser()
    )

    grades = []
    for d in docs:
        try:
            verdict = chain.invoke(
                {"passage": d.page_content[:2000], "question": question}
            ).strip().lower()
            grades.append(verdict.startswith("yes"))
        except Exception:
            grades.append(True)   # fail open — never lose a passage to a grader error
    return grades


def decide(grades: list[bool]) -> str:
    if not grades:
        return "INCORRECT"
    n = sum(grades)
    if n == len(grades):
        return "CORRECT"
    if n == 0:
        return "INCORRECT"
    return "AMBIGUOUS"


# --------------------------------------------------------------------------
# Web fallback
# --------------------------------------------------------------------------

def web_search(query: str, max_results: int = 3) -> list[Document]:
    """DuckDuckGo search, wrapped as Documents so the context formatter is uniform."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("  ddgs not installed — web fallback disabled (pip install ddgs)")
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        print(f"  web search failed: {exc}")
        return []

    return [
        Document(
            page_content=r.get("body", ""),
            metadata={
                "source_type": "web",
                "paper_title": r.get("title", "Web result"),
                "url": r.get("href", ""),
                "page_number": 0,
            },
        )
        for r in results
        if r.get("body")
    ]


# --------------------------------------------------------------------------
# Context formatting with source-type labels
# --------------------------------------------------------------------------

def format_mixed(paper_docs: list[Document], web_docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(paper_docs, 1):
        m = d.metadata
        blocks.append(
            f"[INDEXED PAPER {i}] \"{m.get('paper_title')}\" "
            f"({m.get('authors', '')}, {m.get('year', '')}), page {m.get('page_number')}\n"
            f"{d.page_content.strip()}"
        )
    for j, d in enumerate(web_docs, 1):
        m = d.metadata
        blocks.append(
            f"[WEB {j}] {m.get('paper_title')} — {m.get('url', '')}\n"
            f"{d.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no passages)"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def crag_answer(
    retriever,
    question: str,
    llm=None,
    enable_web: bool = True,
    top_k: int = config.TOP_K,
) -> CRAGResult:
    llm = llm or get_llm()

    retrieved = retriever.invoke(question)[:max(top_k, 5)]
    grades = grade_documents(retrieved, question, llm=llm)
    decision = decide(grades)

    relevant = [d for d, keep in zip(retrieved, grades) if keep][:top_k]

    web_docs: list[Document] = []
    if enable_web and decision in ("AMBIGUOUS", "INCORRECT"):
        web_docs = web_search(question, max_results=3)

    chain = (
        ChatPromptTemplate.from_messages([("system", CRAG_SYSTEM), ("human", CRAG_USER)])
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(
        {"context": format_mixed(relevant, web_docs), "question": question}
    )

    return CRAGResult(
        question=question,
        answer=answer,
        decision=decision,
        n_relevant=sum(grades),
        n_retrieved=len(retrieved),
        used_web=bool(web_docs),
        sources=format_sources(relevant),
        web_sources=[
            {"title": d.metadata.get("paper_title"), "url": d.metadata.get("url")}
            for d in web_docs
        ],
    )


def pretty_print_crag(res: CRAGResult) -> None:
    print("=" * 78)
    print(f"Q: {res.question}")
    print(f"CRAG decision: {res.decision}  "
          f"({res.n_relevant}/{res.n_retrieved} passages graded relevant, "
          f"web fallback: {'yes' if res.used_web else 'no'})")
    print("=" * 78)
    print(res.answer)
    if res.sources:
        print("\nPaper sources:")
        for s in res.sources:
            print(f"  [{s['rank']}] \"{s['paper_title']}\" — page {s['page_number']}")
    if res.web_sources:
        print("\nWeb sources:")
        for w in res.web_sources:
            print(f"  - {w['title']} ({w['url']})")
    print()
