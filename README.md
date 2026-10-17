# 🧠 RAG Demo Series (LangChain + Groq + Streamlit)

A progression of RAG demos built with LangChain (LCEL), Groq open-source models, HuggingFace embeddings, and Streamlit.

## Folders

| Folder | What it shows |
|---|---|
| [`post-1-simple-pdf-rag/`](post-1-simple-pdf-rag/) | Basic PDF RAG: upload → chunk → FAISS → grounded answer via a `pdf_search` tool. No memory, one source, invoke only. |
| [`post-2-lcel-memory/`](post-2-lcel-memory/) | Simple RAG **vs** LCEL + memory, side by side. Multi-source loading (PDF, TXT/MD, web URL), FAISS/Chroma switch, temperature slider, invoke/stream/astream, sources + measured metadata panel. |

## Stack

- **UI / server**: Streamlit
- **LLM**: Groq open-source chat models — `openai/gpt-oss-120b` (default), `qwen/qwen3.8-27b`, `allam-2-7b`
  (Llama models are not available on the current Groq API key; see note below)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **Vector stores**: FAISS or Chroma (switchable at runtime)
- **Loaders**: `PyPDFLoader`, `TextLoader`, `WebBaseLoader`
- **Splitting**: `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), top-k = 4

## Setup (uv, Python 3.14+)

```bash
uv sync                 # creates .venv, installs deps from pyproject.toml
cp .env.example .env    # then add your GROQ_API_KEY (get one free at console.groq.com)
```

## Run

```bash
uv run streamlit run post-2-lcel-memory/app.py     # post-2 demo (recommended)
uv run streamlit run post-1-simple-pdf-rag/app.py  # post-1 demo
```

## Demo script (post-2)

1. **Simple RAG limitation:** upload `NIPS-2017-attention-is-all-you-need-Paper.pdf`, pick **Simple RAG (no memory)**, ask *"What is multi-head attention?"* then *"Why is it better than a single head?"* — the follow-up should fail or be vague, because the retriever only sees the raw follow-up.
2. **LCEL + memory:** clear the chat, switch to **LCEL + memory**, ask the same two questions. The follow-up now works, and the metadata panel shows the rewritten standalone question plus measured rewrite/retrieve latencies.
3. **Vector stores:** switch **FAISS → Chroma** in the sidebar (the index rebuilds) and repeat.
4. **Response modes:** compare **invoke vs stream vs astream** — streaming shows time-to-first-token in the metadata panel.
5. **Temperature:** move the slider (0.0 → 1.0) and ask the same question; the panel logs the temperature actually used.
6. **Multi-source demo:** upload a PDF, paste a web URL together, then ask a question whose answer is only in one of them — confirm the **Retrieved chunks** panel shows the correct loader, file name/URL, and page per chunk. The **Loaded documents** summary shows per-source loader, doc/chunk counts, and measured load time.

## Notes

- Streamlit's file watcher is disabled in `.streamlit/config.toml` (transformers/torchvision scan crash).
- Temp uploads are written to a `tempfile.mkdtemp` directory and removed on process exit.
