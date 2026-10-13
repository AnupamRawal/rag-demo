# PDF RAG Demo (LangChain + Groq + Streamlit)

Upload a PDF and ask questions about it.

## Stack
- **UI / server**: Streamlit
- **LLM**: Groq open-source chat models — `openai/gpt-oss-120b` (default), `qwen/qwen3.8-27b`, `allam-2-7b`
  (Llama models are not available on the current Groq API key; see the note below)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **Vector store**: FAISS (in-memory)
- **PDF parsing**: LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter`
- **Retriever exposed as a LangChain Tool** (`pdf_search`) used to ground the answer

## Setup

This folder is a standalone uv project (Python 3.14+). From inside this folder:

```bash
uv sync          # creates .venv and installs all dependencies from pyproject.toml
```

Add your Groq API key in a `.env` file inside this folder (get one free at
https://console.groq.com):

```
GROQ_API_KEY=gsk_...
```

> generate a new key at
> https://console.groq.com and update `GROQ_API_KEY`, then adjust the model
> list in the sidebar section of `app.py`.

## Run

```bash
cd langchain-rag-demo      # if you are outside the folder
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

Tip: a sample paper (`NIPS-2017-attention-is-all-you-need-Paper.pdf`) is
included — upload it and try asking *"What is the transformer architecture?"*
or *"How does multi-head attention work?"*.

## How it works
1. PDF is uploaded in the sidebar → parsed page-by-page.
2. Text is chunked (1000 chars, 200 overlap) and embedded with MiniLM.
3. Chunks are stored in a FAISS index.
4. A question triggers the `pdf_search` LangChain tool (top-k=4 chunks).
5. Chunks + question go into a prompt → the Groq model answers grounded only in the document, with retrieved chunks shown as sources.

