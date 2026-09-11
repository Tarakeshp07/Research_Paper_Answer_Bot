# Execution Walkthrough

Everything you need to go from an empty folder to a working demo, in order.
Follow it top to bottom the first time.

**Two routes.** Route A is local on your Windows machine — simplest, works
entirely on CPU. Route B is Google Colab — needed if you want GPU speed for the
embedding and reranking steps. You can mix: build the index on Colab, then copy
it down and demo locally.

---

# Route A — Local (Windows)

## Step 0 — Prerequisites

**Python 3.11 is required.** Not 3.12, not 3.13.

This matters more than it usually does. On Python 3.13, several packages in this
stack either have no Windows wheels yet or have versions that explicitly exclude
it, and pip's error messages when that happens are misleading — it reports "no
matching distribution" for a package that exists perfectly well on 3.11.

Check which Python your `python` command actually points at:

```powershell
python --version
```

If it is anything other than 3.11.x, do **not** use `python` for the venv step.
Use `py -3.11` explicitly, as shown below.

If you don't have 3.11 at all, install it from python.org, then verify:

```powershell
py -3.11 --version
```

## Step 1 — Create the virtual environment

```powershell
cd D:\capstone\Codes

py -3.11 -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`. If PowerShell blocks the activation
script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

**Confirm the venv is on 3.11 before installing anything:**

```powershell
python --version
```

This must print `Python 3.11.x`. If it prints anything else, the venv was built
from the wrong interpreter — delete it and redo Step 1:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version
```

## Step 2 — Install dependencies

Install `torch` first, on its own. It is by far the largest package, and
installing the CPU-only build explicitly avoids pulling ~2.5 GB of CUDA
libraries you don't need on a laptop:

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```powershell
pip install -r requirements.txt
```

**Expect 5–15 minutes.** `sentence-transformers`, `chromadb` and `streamlit`
each pull a moderate dependency tree.

Verify the core stack imports:

```powershell
python -c "import langchain, chromadb, sentence_transformers, pymupdf4llm; print('core ok')"
```

### Then the evaluation extras

```powershell
pip install -r requirements-eval.txt
```

RAGAS is deliberately in a separate file. It pins `pydantic` and `datasets`
tightly and is the most common source of dependency conflicts in this stack.
Keeping it out of the core install means a RAGAS resolution failure can't stop
you from building the index and running everything else.

Only notebook §8.2 needs it. If it refuses to install cleanly, skip it for now —
sections 1–7 and 9 are unaffected, and §8.2 will print a clear message instead
of crashing.

### Optional — the Docling parser comparison (notebook §2.2)

```powershell
pip install docling
```

Downloads 1–2 GB of layout models on first use. Skip it if you're short on time
or disk; the notebook handles its absence and that cell just prints a skip note.

## Step 3 — Add your Gemini API key

```powershell
copy .env.example .env
notepad .env
```

Put your key in:

```
GOOGLE_API_KEY=AIza...your_actual_key
```

Get one free at **aistudio.google.com/apikey** if you don't have it handy.

Verify the key actually **works**, not just that it loads:

```powershell
python scripts/check_key.py
```

This makes one tiny API call and diagnoses the result in plain language. Run it
whenever API calls start failing — it takes two seconds and tells you exactly
what's wrong instead of burying the real error under retry spam.

> ### Never put the key anywhere but `.env`
>
> Not in a notebook cell, not in a config file you commit, not in a screenshot.
> Google runs secret scanning across public GitHub and will **permanently
> disable** any key it finds — the error is
> `403 Your API key was reported as leaked`, and no amount of retrying fixes it.
>
> `.env` is in `.gitignore`. Before your first commit in any project, confirm it
> is actually ignored:
>
> ```powershell
> git check-ignore -v .env
> ```
>
> That must print a `.gitignore` line. If it prints nothing, `.env` is **not**
> ignored and will be committed.
>
> If a key does get flagged, `python scripts/find_leaked_key.py` scans your
> files and git history to find where it escaped.

## Step 4 — Build the index

This is the one command that does everything: download → verify → parse →
chunk → embed → index.

```powershell
python scripts/build_index.py
```

**What you should see, in order:**

```
[1/4] Downloading and verifying corpus
Fetching metadata for 14 papers from the arXiv API...
Downloading PDFs: 100%|██████████| 14/14

==============================================================================
CORPUS VERIFICATION
==============================================================================
OK 1706.03762  Attention Is All You Need
OK 1810.04805  BERT: Pre-training of Deep Bidirectional Transformers for...
...
All 14 papers verified.

[2/4] Parsing PDFs with page provenance
Parsing (pymupdf): 100%|██████████| 14/14
Parsed 387 pages from 14 papers -> cached
Page-number check PASSED — 387 pages across 14 papers

[3/4] Chunking
Chunk metadata check PASSED — 1943 chunks, all with page numbers, ids unique
    1943 chunks | mean 812 chars | median 891 | max 1000

[4/4] Indexing
Indexing [bge]: 100%|██████████| 8/8
[bge] indexed 1943 chunks in 94.3s (20.6 chunks/s)
```

**Timing:** 5–12 minutes on CPU. The downloader pauses 3 seconds between papers
— arXiv resets connections under rapid sequential fetching, and being patient
here is cheaper than retrying. This runs once; PDFs and parsed text are cached
permanently afterwards. The first run also pulls the bge model (~440 MB) from
HuggingFace.

**Exact numbers will differ from the sample above** — page and chunk counts
depend on your `pymupdf4llm` version. That's fine. What matters is that both
check lines say PASSED.

### If the download is interrupted

arXiv resets connections under sustained fetching — `ConnectionResetError
(WinError 10054)` partway through is common and is not your fault.

The downloader handles it: five attempts per paper with exponential backoff,
alternating between two arXiv URL forms, streaming to a `.part` file that is
only renamed once complete. A paper that still fails is reported and skipped
rather than aborting the other thirteen.

**Downloads resume.** Already-complete PDFs are skipped on the next run, so if
the script stops partway, just run it again:

```powershell
python scripts/build_index.py
```

To see what you already have without touching the network:

```powershell
python scripts/check_corpus.py
```

```
Downloaded: 9/14
   OK      1706.03762  Transformer           2183 KB
   OK      1810.04805  BERT                   775 KB
   ...
   MISSING 2310.11511  Self-RAG
```

If a specific paper fails repeatedly, download it manually from
`https://arxiv.org/abs/<id>` and save it as `data/papers/<id>.pdf`. The script
will then skip it.

### If verification fails

If any line starts with `!!`, an arXiv id in `src/config.py` points at the wrong
paper. The output shows what was expected vs what arXiv returned. Fix the id in
`CORPUS` and re-run. The script deliberately refuses to continue — indexing an
unverified corpus is how you end up with a confidently wrong citation at the demo.

### Building the Gemini collections

Step 4 above builds only the free local `bge` index. To also build the two
Gemini arms needed for Experiment 1:

```powershell
python scripts/build_index.py --all
```

**This makes ~2,000 embedding API calls.** It is throttled to stay within the
free tier and takes 10–20 minutes. Embeddings are cached to disk, so you pay
this cost exactly once — subsequent runs reuse the cache.

## Step 5 — Run the notebook

```powershell
pip install jupyterlab
jupyter lab
```

Open `notebooks/Research_Paper_Answer_Bot.ipynb` and run cells top to bottom.

Because Step 4 already populated every cache, the ingestion sections complete in
seconds rather than minutes.

**As you go, fill in every blockquote that says "Fill in after running."** There
are seven of them, at the end of sections 2.2, 3.2, 3.3, 4.4, 6.4, 8.3 and 10.
These are the written justifications the rubric grades — the tables alone score
about half of what a table plus its reasoning scores.

## Step 6 — Launch the demo UI

```powershell
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

First load takes 30–60 seconds while chunks load into memory for BM25. After
that, switching strategies in the sidebar is instant.

**Five queries to demo, in this order** — they cover every capability the viva
asks about:

1. *"What is scaled dot-product attention and why is the dot product scaled?"*
   → single-hop lookup, clean citation
2. *"How do Chinchilla's scaling conclusions differ from Kaplan's?"*
   → cross-paper retrieval, cites two papers
3. *"In LoRA, which weights are frozen and what is injected instead?"*
   → precise technical detail
4. *"Which of those steps does DPO eliminate?"* (as a follow-up to a question
   about InstructGPT) → conversational memory; the rewritten question is shown
5. *"What is the context window of Gemini 2.5 Flash?"* with CRAG toggled on
   → out-of-corpus, triggers the web fallback and labels it

Switch the retrieval strategy in the sidebar between queries to show Experiment 2
live. That turns a static table into something the examiner watches happen.

---

# Route B — Google Colab

## Step 1 — Put the project on Drive

### Zip it properly first

Do **not** zip the whole `Codes` folder. It contains `.venv`, which is several
hundred MB of Windows-compiled binaries that cannot run on Colab's Linux. A
full-folder zip comes out around 800 MB; a correct one is under 1 MB.

From PowerShell:

```powershell
cd D:\capstone
Compress-Archive -Path `
  Codes\src, Codes\app, Codes\scripts, Codes\notebooks, `
  Codes\data\eval, Codes\requirements.txt, Codes\requirements-eval.txt, `
  Codes\README.md, Codes\WALKTHROUGH.md `
  -DestinationPath Codes_slim.zip -Force
```

Optionally add `Codes\data\papers, Codes\data\cache` (~30–40 MB) to carry the
downloaded PDFs and parsed text across, saving ~10 minutes of re-download and
re-parse on Colab.

### Upload and extract

Upload `Codes_slim.zip` to a Drive folder — say `MyDrive/capstone_rag`. Then in
a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/capstone_rag'

# -o overwrites; the -x patterns are belt-and-braces in case .venv got zipped
!cd {BASE} && unzip -q -o *.zip -x "*/.venv/*" ".venv/*" "*/__pycache__/*"

# Find where src/ actually landed — zip layouts vary
!find {BASE} -name "config.py" -path "*/src/*"
```

Set `PROJECT_DIR` to the directory **containing** `src/`. If the find printed
`/content/drive/MyDrive/capstone_rag/Codes/src/config.py`, then:

```python
PROJECT_DIR = '/content/drive/MyDrive/capstone_rag/Codes'
```

Verify before continuing:

```python
import os
assert os.path.isdir(f'{PROJECT_DIR}/src'), f'src/ not found under {PROJECT_DIR}'
print('project root ok:', PROJECT_DIR)
```

### If unzip says "cannot find or open"

You're in the wrong directory. `unzip` looks relative to wherever `cd` put you,
so a zip sitting inside `capstone_rag/` is not found by a command run from
`MyDrive/`. Check what's actually there:

```python
!ls -la /content/drive/MyDrive/capstone_rag
```

Then `cd` to the folder the zip is actually in.

## Step 2 — Add the API key as a Colab secret

Click the **key icon** in the left sidebar → **Add new secret**:

- Name: `GOOGLE_API_KEY`
- Value: your key
- Toggle **Notebook access** on

This keeps the key out of the notebook, which matters because you're submitting
the notebook.

## Step 3 — Enable GPU

**Runtime → Change runtime type → T4 GPU → Save**

## Step 4 — Run the notebook

Open `notebooks/Research_Paper_Answer_Bot.ipynb` from Drive and run top to bottom.
Cell 1 handles the mount and dependency install; make sure `PROJECT_DIR` points at
`/content/drive/MyDrive/capstone_rag`.

Everything (PDFs, caches, Chroma index) is written to Drive, so a runtime
disconnect never costs you a rebuild.

**On GPU, indexing runs roughly 8–10x faster than CPU.**

## Step 5 — Streamlit on Colab

Streamlit needs a tunnel to be reachable:

```python
!npm install -g localtunnel
!streamlit run app/streamlit_app.py &>/content/log.txt &
!npx localtunnel --port 8501
```

Click the printed URL. The tunnel password is the output of:

```python
!curl https://loca.lt/mytunnelpassword
```

**For the live viva, prefer running the app locally.** Tunnels drop, ask for
passwords, and show interstitial pages — exactly what you don't want mid-demo.
Copy `data/chroma/` and `data/cache/` down from Drive to your Windows folder and
run Step 6 of Route A. Query-time embedding on CPU is ~50 ms, entirely fine.

---

# Command reference

| Command | What it does |
|---|---|
| `python scripts/build_index.py` | Full pipeline, bge index only (free) |
| `python scripts/build_index.py --all` | Also builds both Gemini indexes (uses quota) |
| `python scripts/build_index.py --rebuild` | Drops and rebuilds indexes |
| `python scripts/build_index.py --no-cache` | Ignores the parse cache, re-parses PDFs |
| `python -m src.download` | Just download and verify the corpus |
| `python scripts/check_corpus.py` | Show which PDFs are already downloaded (offline) |
| `python scripts/check_key.py` | Test the Gemini API key and diagnose failures |
| `python scripts/find_leaked_key.py` | Scan files and git history for exposed API keys |
| `streamlit run app/streamlit_app.py` | Launch the demo UI |

---

# Troubleshooting

### `403 Your API key was reported as leaked`

**The key is permanently dead.** Google's secret scanner found it published
somewhere public and disabled it. Retrying will never succeed.

1. Revoke it at https://aistudio.google.com/apikey — delete that key.
2. Create a new one and put it in `.env`.
3. Find the leak: `python scripts/find_leaked_key.py`
   Deleting a file does not remove a key from git history, and history is what
   gets pushed. If you don't fix the source, the new key gets flagged too.
4. Verify the new key: `python scripts/check_key.py`

### `GOOGLE_API_KEY not found`
Local: `.env` must sit in `D:\capstone\Codes\` (not in `src/`) and contain
`GOOGLE_API_KEY=...` with no quotes and no spaces around the `=`.
Colab: the secret needs **Notebook access** toggled on — adding it isn't enough.

### `429 ResourceExhausted` during indexing
You've hit the Gemini free-tier rate limit. The embedder already retries with
exponential backoff; if it still fails, raise `GEMINI_EMBED_SLEEP` in
`src/config.py` from `1.0` to `3.0` and re-run. Cached embeddings are kept, so it
resumes rather than starting over.

### `ModuleNotFoundError: No module named 'src'`
You're running from the wrong directory. All commands run from
`D:\capstone\Codes`, not from inside `src/` or `notebooks/`.

### Index exists but returns nothing
The collection was built with a different embedding model than you're querying
with. Vector dimensions must match. Rebuild:
```powershell
python scripts/build_index.py --rebuild
```

### `AssertionError` in the page-number check
Parsing lost page provenance — likely a `pymupdf4llm` version change. Check
`parse_pymupdf()` still receives one dict per page. Do not work around this by
disabling the assertion; page numbers are graded in four places.

### Chunk counts changed after re-running
You changed `CHUNK_SIZE` or `CHUNK_OVERLAP` in `src/config.py`. Chunk ids are
derived from position, so existing indexes are now stale. Rebuild with
`--rebuild`.

### Streamlit shows a Chroma error on first query
The index hasn't been built. Run `python scripts/build_index.py` first.

### `torch` install fails or takes forever
Install the CPU-only build first, then the rest:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Docling import errors
It's optional. Uninstall it (`pip uninstall docling`) — the notebook skips that
comparison cell cleanly and everything else works.

---

# Order of work for the 30-day window

1. **Today** — Steps 1–4 of Route A. Confirm both check lines say PASSED.
   Nothing else matters until page numbers are verified.
2. **Next** — run the notebook through §3 and write the EDA observations while
   the data is fresh in your mind.
3. **Then** — `--all` to build the Gemini indexes, run Experiment 1, write the
   embedding decision.
4. **Then** — Experiment 2, the failure analysis, and the retrieval decision.
5. **Then** — §7 and §8: the chain, RAGAS, refusal behaviour.
6. **Then** — §9 stretch goals, then the UI demo rehearsal.
7. **Last** — the 12-slide deck, then a full fresh-kernel run of the notebook
   before you zip and submit.

The single most common way to lose marks here is leaving the "Fill in after
running" blockquotes empty. The code produces the numbers; the marks come from
you explaining what the numbers mean.
