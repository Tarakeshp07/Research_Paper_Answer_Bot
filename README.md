# Research Paper Answer Bot

A Retrieval-Augmented Generation system that answers technical questions about
14 seminal GenAI research papers, citing the paper title and page number behind
every claim.

**GenAI Pinnacle Plus Capstone — Analytics Vidhya**

---

## What it does

Ask a question about the transformer architecture, scaling laws, LoRA, RLHF or
any of the indexed papers. The system retrieves the most relevant passages,
generates a grounded answer, and shows the top-3 supporting passages with paper
title and page number. If the corpus cannot answer, it says so rather than
guessing.

## Architecture

```
14 arXiv PDFs
      |
      v
  Docling / pymupdf4llm          page-tagged text
      |
      v
  Recursive chunking             ~2,000 chunks, each with
      |                          {paper_title, page_number, section}
      v
  +---------------------------+
  |  ChromaDB (cosine space)  |  3 collections, one per embedding model
  |  bge-base | gemini | g768 |
  +---------------------------+
      |
      v
  Retrieval:  dense | MMR | hybrid BM25+RRF | + cross-encoder rerank
      |
      v
  [CRAG] grade passages -> web fallback if coverage is poor
      |
      v
  Gemini 2.5 Flash + grounding prompt
      |
      v
  Answer with [Paper Title, p.N] citations + top-3 passages
```

## Project layout

```
Codes/
├── notebooks/
│   └── Research_Paper_Answer_Bot.ipynb   <- the graded deliverable
├── src/
│   ├── config.py          paths, model ids, corpus definition
│   ├── download.py        arXiv fetch + title verification
│   ├── parse.py           PDF -> page-tagged text (2 parsers)
│   ├── chunking.py        recursive + semantic strategies
│   ├── embeddings.py      bge / gemini behind one interface, disk-cached
│   ├── indexing.py        Chroma build, cosine space, true similarity
│   ├── retrievers.py      dense | mmr | hybrid | rerank
│   ├── rag_chain.py       LCEL chain, prompt, citations, memory
│   ├── evaluate.py        hit@k / MRR / nDCG + RAGAS
│   └── crag.py            corrective RAG with web fallback
├── app/
│   └── streamlit_app.py   demo UI
├── scripts/
│   └── build_index.py     one-shot: download -> parse -> chunk -> index
├── data/
│   ├── papers/            downloaded PDFs
│   ├── cache/             parsed docs + embedding cache
│   ├── chroma/            vector store
│   └── eval/questions.json  20 labelled test questions
└── results/               experiment tables (CSV)
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini API key
python scripts/build_index.py
streamlit run app/streamlit_app.py
```

Full instructions, expected timings and troubleshooting: **WALKTHROUGH.md**

## Deliverables against the brief

| Requirement | Where |
|---|---|
| Corpus indexed in a vector DB | `src/indexing.py`, notebook §5 |
| ≥2 embedding models compared | Experiment 1, notebook §4 |
| ≥2 retrieval strategies compared | Experiment 2, notebook §6 |
| Complete RAG chain | `src/rag_chain.py`, notebook §7 |
| Top-3 passages with title + page | every answer |
| ≥10 test questions + evaluation | `data/eval/questions.json`, notebook §8 |
| Stretch goal 1 — chat memory | `rag_chain.build_conversational_chain` |
| Stretch goal 2 — UI | `app/streamlit_app.py` |
| Stretch goal 3 — CRAG | `src/crag.py` |

## Corpus

Transformer · BERT · GPT-3 · RAG · DPR · Kaplan Scaling Laws · Chinchilla ·
Chain-of-Thought · InstructGPT · DPO · LoRA · ReAct · Self-RAG · Lost in the Middle
