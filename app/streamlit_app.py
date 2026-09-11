"""
Stretch goal 2 — Streamlit UI.

Run from the PROJECT ROOT (the folder containing src/ and app/):

    streamlit run app/streamlit_app.py

Demo value: the viva asks for at least 5 live queries with supporting passages
shown. The sidebar lets you switch retrieval strategy live, which turns
Experiment 2 from a table in a notebook into something the examiner can watch
change in real time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                                    # noqa: E402
from src.chunking import chunk_recursive                  # noqa: E402
from src.crag import crag_answer, CRAGResult              # noqa: E402
from src.download import download_corpus                  # noqa: E402
from src.embeddings import EMBEDDING_SPECS                # noqa: E402
from src.indexing import load_index                       # noqa: E402
from src.parse import parse_corpus                        # noqa: E402
from src.rag_chain import (                               # noqa: E402
    ask,
    build_conversational_chain,
    build_rag_chain,
)
from src.retrievers import STRATEGY_LABELS, get_retriever  # noqa: E402

st.set_page_config(page_title="Research Paper Answer Bot", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading chunks (needed for BM25)...")
def load_chunks():
    papers = download_corpus()
    pages = parse_corpus(papers, parser="pymupdf", use_cache=True)
    return chunk_recursive(pages)


@st.cache_resource(show_spinner="Opening vector index...")
def load_store(embedder_name: str):
    return load_index(embedder_name)


@st.cache_resource(show_spinner="Building retriever...")
def build_retriever(embedder_name: str, strategy: str, k: int):
    store = load_store(embedder_name)
    docs = load_chunks() if strategy in ("hybrid", "rerank") else None
    return get_retriever(strategy, store, docs, k=k)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Configuration")

embedder_name = st.sidebar.selectbox(
    "Embedding model",
    options=list(EMBEDDING_SPECS.keys()),
    format_func=lambda k: EMBEDDING_SPECS[k]["label"],
    index=0,
)

strategy = st.sidebar.selectbox(
    "Retrieval strategy",
    options=list(STRATEGY_LABELS.keys()),
    format_func=lambda k: STRATEGY_LABELS[k],
    index=3,
)

top_k = st.sidebar.slider("Passages to retrieve (k)", 1, 8, config.TOP_K)

st.sidebar.divider()
st.sidebar.caption("Stretch goals")
use_memory = st.sidebar.toggle("Conversational memory", value=True)
use_crag = st.sidebar.toggle("Corrective RAG (web fallback)", value=False)

if use_crag:
    st.sidebar.caption("CRAG grades each passage and falls back to web search "
                       "when the corpus lacks coverage. Web-sourced content is "
                       "labelled in the answer.")

st.sidebar.divider()
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

with st.sidebar.expander("Indexed corpus"):
    for p in config.CORPUS:
        st.caption(f"{p['short']} — arXiv:{p['arxiv_id']}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("Research Paper Answer Bot")
st.caption(
    "Grounded question answering over 14 seminal GenAI papers. "
    "Every claim is cited with paper title and page number."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict], web_sources: list[dict] | None = None):
    if sources:
        st.markdown("**Supporting passages**")
        for s in sources:
            label = f"[{s['rank']}] {s['paper_title']} — page {s['page_number']}"
            if s.get("section"):
                label += f" · {s['section']}"
            with st.expander(label):
                st.write(s["excerpt"])
                st.caption(f"chunk_id: `{s['chunk_id']}`")
    if web_sources:
        st.markdown("**Web sources (CRAG fallback)**")
        for w in web_sources:
            st.markdown(f"- [{w['title']}]({w['url']})")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("standalone") and msg["standalone"] != msg.get("original"):
                st.caption(f"Rewritten for retrieval: _{msg['standalone']}_")
            if msg.get("decision"):
                st.caption(f"CRAG decision: **{msg['decision']}** · "
                           f"{msg['n_relevant']}/{msg['n_retrieved']} passages relevant")
            render_sources(msg.get("sources", []), msg.get("web_sources"))


question = st.chat_input("Ask about the indexed papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                retriever = build_retriever(embedder_name, strategy, top_k)

                if use_crag:
                    res: CRAGResult = crag_answer(retriever, question, top_k=top_k)
                    payload = {
                        "role": "assistant",
                        "content": res.answer,
                        "sources": res.sources,
                        "web_sources": res.web_sources,
                        "decision": res.decision,
                        "n_relevant": res.n_relevant,
                        "n_retrieved": res.n_retrieved,
                    }

                elif use_memory:
                    from langchain_core.messages import AIMessage, HumanMessage

                    history = []
                    for m in st.session_state.messages[:-1]:
                        cls = HumanMessage if m["role"] == "user" else AIMessage
                        history.append(cls(content=m["content"]))

                    chain = build_conversational_chain(retriever)
                    res = chain.invoke({"question": question, "chat_history": history})
                    payload = {
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res["sources"],
                        "standalone": res["standalone_question"],
                        "original": question,
                    }

                else:
                    chain = build_rag_chain(retriever)
                    res = ask(chain, question)
                    payload = {
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res["sources"],
                    }

                st.markdown(payload["content"])
                if payload.get("standalone") and payload["standalone"] != question:
                    st.caption(f"Rewritten for retrieval: _{payload['standalone']}_")
                if payload.get("decision"):
                    st.caption(f"CRAG decision: **{payload['decision']}** · "
                               f"{payload['n_relevant']}/{payload['n_retrieved']} passages relevant")
                render_sources(payload.get("sources", []), payload.get("web_sources"))

                st.session_state.messages.append(payload)

            except Exception as exc:
                from src.rag_chain import classify_api_error

                diagnosis = classify_api_error(exc)
                if diagnosis:
                    st.error("API key problem — retrying will not help")
                    st.markdown(diagnosis)
                    st.code("python scripts/check_key.py", language="powershell")
                else:
                    st.error(f"{type(exc).__name__}: {exc}")
                    st.caption(
                        "Common causes: the index has not been built yet "
                        "(run `python scripts/build_index.py`), or GOOGLE_API_KEY is missing."
                    )
