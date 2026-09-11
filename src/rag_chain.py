"""
Step 6 — The RAG pipeline.

Built with LCEL (LangChain Expression Language) because the viva allocates
10 marks to LangChain component understanding, and LCEL's `|` composition is
the thing you will be asked to explain.

Three design choices carry most of the grounding quality:

 1. Citation metadata goes INTO the context block, pre-formatted. We never ask
    the model to remember which paper a passage came from — it reads the label
    sitting directly above the text. Hallucinated citations mostly come from
    asking a model to recall provenance it was never shown.

 2. Temperature 0. This is extraction and synthesis, not writing.

 3. An explicit refusal string. "Say I don't know" is not enough; giving the
    model the EXACT sentence to emit makes refusals detectable in evaluation.
"""

from __future__ import annotations

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from . import config

REFUSAL = "I don't have enough information in the indexed papers to answer that."

SYSTEM_PROMPT = f"""You are a research assistant answering questions about AI research papers.

RULES:
1. Answer ONLY from the provided context. Never use outside knowledge, even if \
you are confident it is correct.
2. If the context does not contain enough information, reply with exactly this \
sentence and nothing else: "{REFUSAL}"
3. Cite every factual claim inline using the format [Paper Title, p.N], taking \
the title and page number from the source header above each passage.
4. If sources disagree, say so explicitly and cite both.
5. Be concise and technical. Do not pad the answer with restatements of the question.
"""

USER_PROMPT = """Context passages:

{context}

---

Question: {question}

Answer (with inline [Paper Title, p.N] citations):"""


# --------------------------------------------------------------------------
# Context formatting — where citation fidelity is won
# --------------------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    """
    Render retrieved chunks into a context block where provenance is impossible
    to miss. Each passage is preceded by a header the model can copy verbatim.
    """
    if not docs:
        return "(no passages retrieved)"

    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = (
            f"[Source {i}] \"{m.get('paper_title', 'Unknown')}\" "
            f"({m.get('authors', 'Unknown')}, {m.get('year', 'n.d.')}), "
            f"page {m.get('page_number', '?')}"
        )
        if m.get("section"):
            header += f", section: {m['section']}"
        blocks.append(f"{header}\n{d.page_content.strip()}")

    return "\n\n---\n\n".join(blocks)


def format_sources(docs: list[Document]) -> list[dict]:
    """Structured source list for the UI and for the notebook's results tables."""
    return [
        {
            "rank": i,
            "paper_title": d.metadata.get("paper_title", "Unknown"),
            "paper_id": d.metadata.get("paper_id", ""),
            "page_number": d.metadata.get("page_number"),
            "section": d.metadata.get("section", ""),
            "chunk_id": d.metadata.get("chunk_id", ""),
            "excerpt": d.page_content.strip()[:400],
        }
        for i, d in enumerate(docs, 1)
    ]


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

class ApiKeyError(RuntimeError):
    """Raised for auth failures that retrying can never fix."""


def classify_api_error(exc: Exception) -> str | None:
    """
    Distinguish permanent auth failures from transient ones.

    This matters because the default client retries everything with exponential
    backoff. A revoked key then produces two minutes of retry spam before
    surfacing an error that was never going to change.
    """
    msg = str(exc).lower()
    if "leaked" in msg:
        return (
            "This API key was flagged by Google as PUBLICLY LEAKED and is permanently "
            "disabled. Revoke it at https://aistudio.google.com/apikey, create a new "
            "one, and run `python scripts/find_leaked_key.py` to find the source — "
            "otherwise the replacement gets flagged too."
        )
    if "api key not valid" in msg or "api_key_invalid" in msg:
        return (
            "Google does not recognise this API key. Check .env for stray quotes or "
            "spaces, or create a new key at https://aistudio.google.com/apikey"
        )
    if "permission" in msg and "denied" in msg:
        return (
            "Permission denied. The key exists but cannot call this model — check the "
            "Generative Language API is enabled and the key has no API restrictions."
        )
    return None


def get_llm(
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    max_retries: int = 2,
):
    """
    Gemini chat model.

    max_retries defaults to 2 rather than the client default (6). Transient 503s
    are worth one or two retries; auth failures are not worth any, and the long
    default turns a dead key into minutes of backoff before the real error shows.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    config.get_google_api_key()
    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, max_retries=max_retries
    )


def verify_llm() -> None:
    """
    One cheap call to confirm the key works. Call before an expensive run so a
    dead key fails in two seconds rather than halfway through an experiment.
    """
    try:
        get_llm(max_retries=0).invoke("Reply with the single word: ok")
    except Exception as exc:
        diagnosis = classify_api_error(exc)
        if diagnosis:
            raise ApiKeyError(diagnosis) from exc
        raise


# --------------------------------------------------------------------------
# Chain construction
# --------------------------------------------------------------------------

def build_rag_chain(retriever, llm=None):
    """
    Single-turn RAG chain.

    Composition, read left to right:
        {context: retriever | format_docs, question: passthrough}
            | prompt | llm | parser

    RunnableParallel runs retrieval and question passthrough concurrently, then
    hands both to the prompt. `.assign()` keeps the raw documents alongside the
    generated answer so the caller can show sources without retrieving twice.
    """
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    retrieve = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    )

    answer = (
        RunnableParallel(
            context=itemgetter("docs") | RunnableLambda(format_docs),
            question=itemgetter("question"),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieve | RunnableParallel(
        answer=answer,
        docs=itemgetter("docs"),
        question=itemgetter("question"),
    )


def ask(chain, question: str) -> dict:
    """Invoke the chain and return a tidy result dict."""
    result = chain.invoke(question)
    docs = result["docs"]
    return {
        "question": question,
        "answer": result["answer"],
        "sources": format_sources(docs),
        "n_sources": len(docs),
        "refused": REFUSAL.lower() in result["answer"].lower(),
        "_docs": docs,
    }


def pretty_print(result: dict) -> None:
    """Console rendering for notebook demos — answer, then top-3 with pages."""
    print("=" * 78)
    print(f"Q: {result['question']}")
    print("=" * 78)
    print(result["answer"])
    print("\n" + "-" * 78)
    print(f"TOP {len(result['sources'])} SUPPORTING PASSAGES")
    print("-" * 78)
    for s in result["sources"]:
        print(f"\n[{s['rank']}] \"{s['paper_title']}\" — page {s['page_number']}"
              + (f" — {s['section']}" if s["section"] else ""))
        print(f"    {s['excerpt'][:280]}...")
    print()


# --------------------------------------------------------------------------
# Stretch goal 1 — conversational memory
# --------------------------------------------------------------------------

CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, \
rewrite the follow-up as a STANDALONE question that makes sense without the \
conversation history. Resolve every pronoun and implicit reference.

If the follow-up is already standalone, return it unchanged. \
Return ONLY the rewritten question, nothing else."""


def build_conversational_chain(retriever, llm=None):
    """
    History-aware RAG.

    The problem: "what about the second one?" embeds to nothing useful. Sending
    it straight to the retriever returns garbage regardless of how good the
    retriever is.

    The fix: a cheap LLM call rewrites the follow-up into a standalone question
    using chat history, and THAT is what gets retrieved. Retrieval quality is
    restored, and the rewritten question is inspectable — which makes the
    behaviour easy to demo and easy to debug.
    """
    llm = llm or get_llm()

    condense = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    def _standalone(inputs: dict) -> str:
        if not inputs.get("chat_history"):
            return inputs["question"]
        return (condense | llm | StrOutputParser()).invoke(inputs).strip()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    def _run(inputs: dict) -> dict:
        standalone = _standalone(inputs)
        docs = retriever.invoke(standalone)
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "question": standalone}
        )
        return {
            "question": inputs["question"],
            "standalone_question": standalone,
            "answer": answer,
            "sources": format_sources(docs),
            "n_sources": len(docs),
            "refused": REFUSAL.lower() in answer.lower(),
            "_docs": docs,
        }

    return RunnableLambda(_run)
