# 🧠 RAG Demo Series (LangChain + Groq + Streamlit)

A progression of RAG demos built with LangChain (LCEL), Groq open-source models, HuggingFace embeddings, and Streamlit.

## Project structure

```
langchain-rag-demo/
├── .env                        # your GROQ_API_KEY (gitignored)
├── .env.example                # template — copy to .env
├── .python-version             # 3.14
├── .streamlit/config.toml      # global Streamlit config (file watcher disabled)
├── pyproject.toml              # uv project config + pinned dependencies
├── uv.lock                     # lockfile
├── NIPS-2017-attention-is-all-you-need-Paper.pdf   # sample test PDF
└── src/
    ├── post-1-simple-pdf-rag/
    │   └── app.py              # single-file PDF RAG demo
    └── post-2-lcel-memory/
        ├── app.py              # main Streamlit app (UI, sidebar, chat, panels)
        ├── rag_chains.py       # LCEL chains: simple RAG + memory RAG
        ├── stores.py           # loaders, embeddings, FAISS/Chroma builders
        └── .streamlit/config.toml
```

## What each demo shows

| Folder | Purpose |
|---|---|
| `src/post-1-simple-pdf-rag/` | Basic PDF RAG: upload → chunk → FAISS → grounded answer via a `pdf_search` tool. No memory, one source, invoke only. |
| `src/post-2-lcel-memory/` | Simple RAG **vs** LCEL + memory, side by side. Multi-source loading (PDF, TXT/MD, web URL), FAISS/Chroma switch, temperature slider, invoke/stream/astream response modes, sources + measured metadata panel. |

## Stack

- **UI / server**: Streamlit
- **LLM**: Groq open-source chat models — `openai/gpt-oss-120b` (default), `qwen/qwen3.8-27b`, `allam-2-7b`
  (Llama models are not available on the current Groq API key; generate a new one at console.groq.com if needed)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **Vector stores**: FAISS or Chroma (switchable at runtime in post-2)
- **Loaders**: `PyPDFLoader`, `TextLoader`, `WebBaseLoader`
- **Splitting**: `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), top-k = 4

## Setup (uv, Python 3.14+)

```bash
uv sync                 # creates .venv, installs deps from pyproject.toml
cp .env.example .env    # then add your GROQ_API_KEY (get one free at console.groq.com)
```

## Run

```bash
uv run streamlit run src/post-2-lcel-memory/app.py     # post-2 demo (recommended)
uv run streamlit run src/post-1-simple-pdf-rag/app.py  # post-1 demo
```

## Demo script (post-2)

1. **Simple RAG limitation:** upload `NIPS-2017-attention-is-all-you-need-Paper.pdf`, pick **Simple RAG (no memory)**, ask *"What is multi-head attention?"* then *"Why is it better than a single head?"* — the follow-up should fail or be vague, because the retriever only sees the raw follow-up.
2. **LCEL + memory:** clear the chat, switch to **LCEL + memory**, ask the same two questions. The follow-up now works, and the metadata panel shows the rewritten standalone question plus measured rewrite/retrieve latencies.
3. **Vector stores:** switch **FAISS → Chroma** in the sidebar (the index rebuilds) and repeat.
4. **Response modes:** compare **invoke vs stream vs astream** — streaming shows time-to-first-token in the metadata panel.
5. **Temperature:** move the slider (0.0 → 1.0) and ask the same question; the panel logs the temperature actually used.
6. **Multi-source demo:** upload a PDF and paste a web URL together, then ask a question whose answer is only in one of them — confirm the **Retrieved chunks** panel shows the correct loader, file name/URL, and page per chunk. The **Loaded documents** summary shows per-source loader, doc/chunk counts, and measured load time.

## Demo script (post-1)

1. Upload the sample PDF.
2. Ask *"What is multi-head attention?"*.
3. Follow up with *"Why is it better than a single head?"* — with no memory, the follow-up should fail or be vague.

## Notes

- Streamlit's file watcher is disabled in `.streamlit/config.toml` (transformers/torchvision scan crash on this setup).
- Temp uploads are written to a `tempfile.mkdtemp` directory and removed on process exit.
- All metadata-panel numbers (latency, time-to-first-token, token usage, load times) are measured at runtime — nothing is estimated.
