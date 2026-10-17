"""
post-2: LCEL + memory RAG demo, side by side with Simple RAG.

Shows the limitation of Simple RAG (no history) vs a conversational LCEL
chain with question rewriting and in-memory message history, plus
multi-source loading, FAISS/Chroma switching, and invoke/stream/astream.
"""

import asyncio
import os
import shutil
import tempfile
import time

# Must be set before transformers/torch are imported
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chains import (
    build_memory_chain,
    build_simple_chain,
    get_history_factory,
    make_llm,
)
from stores import (
    SourceSpec,
    build_vector_store,
    get_embeddings,
    load_source,
)

load_dotenv()

st.set_page_config(page_title="LCEL + Memory RAG Demo", page_icon="🧠", layout="wide")
st.title("🧠 post-2: LCEL + Memory RAG")
st.caption(
    "Simple RAG vs LCEL + memory · FAISS / Chroma · multi-source loading · "
    "invoke / stream / astream · Groq open-source models"
)

MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]
TMP_DIR = tempfile.mkdtemp(prefix="rag_demo_")


def save_upload(uploaded_file) -> str:
    """Persist an UploadedFile to a temp path (cleaned up on exit)."""
    path = os.path.join(TMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


with st.sidebar:
    st.header("1. Document sources")

    pdf_file = st.file_uploader("PDF (PyPDFLoader)", type=["pdf"])
    text_file = st.file_uploader(
        "Text / Markdown (TextLoader)", type=["txt", "md"]
    )
    url_input = st.text_input("Web URL (WebBaseLoader)", placeholder="https://...")

    specs: list[SourceSpec] = []
    if pdf_file:
        specs.append(
            SourceSpec("pdf", "PyPDFLoader", save_upload(pdf_file), pdf_file.name)
        )
    if text_file:
        specs.append(
            SourceSpec("text", "TextLoader", save_upload(text_file), text_file.name)
        )
    if url_input.strip():
        specs.append(
            SourceSpec("url", "WebBaseLoader", url_input.strip(),
                       url_input.strip())
        )

    st.divider()
    st.header("2. Settings")
    store_type = st.radio("Vector store", ["FAISS", "Chroma"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    groq_model = st.selectbox("Groq model", MODELS)
    response_mode = st.radio("Response mode", ["invoke", "stream", "astream"])
    rag_mode = st.radio(
        "RAG mode",
        ["Simple RAG (no memory)", "LCEL + memory"],
        help="Simple RAG ignores chat history; LCEL + memory rewrites "
        "follow-ups into standalone questions using the history.",
    )

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.history_store = {}
        st.rerun()

    if "summaries" in st.session_state:
        st.divider()
        st.subheader("📄 Loaded documents")
        for s in st.session_state.summaries:
            if s["error"]:
                st.error(f"❌ {s['source']} ({s['loader']}): {s['error']}")
            else:
                st.markdown(
                    f"**{s['source']}** · {s['loader']} · "
                    f"{s['docs']} docs → {s['chunks']} chunks · "
                    f"loaded in {s['load_s']}s"
                )
        st.caption(f"Index built in {st.session_state.index_seconds}s")

# Rebuild the index only when sources/store change (every chat message is a rerun)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_store" not in st.session_state:
    st.session_state.history_store = {}  # session_id -> ChatMessageHistory

embeddings = get_embeddings()

# Fingerprint: same sources + same store => keep the cached index instead
# of rebuilding on every Streamlit rerun (every chat message is a rerun).
fingerprint = (
    tuple(
        (s.source_type, s.display_name, s.path_or_url) for s in specs
    ),
    store_type,
)

if not specs:
    st.info("👈 Add at least one document source in the sidebar to begin.")
    st.stop()

if (
    st.session_state.get("index_fingerprint") != fingerprint
    or "vectorstore" not in st.session_state
):
    with st.spinner("Loading and indexing sources..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks, summaries = [], []
        for spec in specs:  # per-source errors are caught inside load_source
            res = load_source(spec)
            if res.error:
                summaries.append(
                    {
                        "source": spec.display_name,
                        "loader": spec.loader_name,
                        "docs": 0,
                        "chunks": 0,
                        "load_s": None,
                        "error": res.error,
                    }
                )
                continue
            chunks = splitter.split_documents(res.docs)
            all_chunks.extend(chunks)
            summaries.append(
                {
                    "source": spec.display_name,
                    "loader": spec.loader_name,
                    "docs": len(res.docs),
                    "chunks": len(chunks),
                    "load_s": round(res.load_seconds, 3),
                    "error": None,
                }
            )

        if not all_chunks:
            st.error("No text could be indexed from any source.")
            st.stop()

        t0 = time.perf_counter()
        st.session_state.vectorstore = build_vector_store(
            store_type, all_chunks, embeddings
        )
        st.session_state.index_fingerprint = fingerprint
        st.session_state.summaries = summaries
        st.session_state.index_seconds = round(time.perf_counter() - t0, 3)
        st.session_state.messages = []  # sources changed -> reset chat
        st.session_state.history_store = {}

vectorstore = st.session_state.vectorstore

# --------------------------------------------------------------------------
# Chat UI (st.chat_message + st.chat_input, history in session_state)
# --------------------------------------------------------------------------
import json

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = make_llm(groq_model, temperature)
history_factory = get_history_factory(st.session_state.history_store)
SESSION_ID = "streamlit-session"  # store lives in st.session_state


def render_sources(docs):
    """Sources panel: loader, source name and page per retrieved chunk."""
    with st.expander(f"🔎 Retrieved chunks ({len(docs)})"):
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page")
            page_str = f", page {int(page) + 1}" if isinstance(page, int) else ""
            st.markdown(
                f"**{i}.** `{doc.metadata.get('loader', '?')}` · "
                f"`{doc.metadata.get('source_name', '?')}`{page_str}"
            )
            st.text(doc.page_content[:500])


def render_metadata(meta: dict):
    """Metadata panel — only measured/runtime values, never fabricated."""
    with st.expander("📊 Metadata"):
        st.markdown(
            f"RAG mode: **{meta['rag_mode']}** · Store: **{meta['store']}** · "
            f"Temp: **{meta['temperature']}** · Mode: **{meta['response_mode']}**"
        )
        st.markdown(f"- Total latency: **{meta['total_seconds']:.3f}s**")
        if meta.get("ttft_seconds") is not None:
            st.markdown(f"- Time to first token: **{meta['ttft_seconds']:.3f}s**")
        st.markdown(
            f"- Token usage: `{json.dumps(meta['usage'])}`"
            if meta.get("usage") is not None
            else "- Token usage: not returned"
        )
        if meta.get("standalone_question") is not None:
            st.markdown(
                f"- Rewritten standalone question: "
                f"**{meta['standalone_question']}** "
                f"(rewrite: {meta['rewrite_seconds']:.3f}s, "
                f"retrieve: {meta['retrieve_seconds']:.3f}s)"
            )


# Render previous chat turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            render_sources(msg["chunks"])
        if msg.get("meta"):
            render_metadata(msg["meta"])

# Chat input
if prompt := st.chat_input("Ask a question about your sources..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        meta = {
            "rag_mode": rag_mode,
            "store": store_type,
            "temperature": temperature,
            "response_mode": response_mode,
            "usage": None,
            "standalone_question": None,
        }
        sink: dict = {}  # rag_chains writes measured step data here

        try:
            # Usage callback captures token usage from the LLM call(s)
            # regardless of chain output type (StrOutputParser strips
            # usage_metadata from the final value).
            usage_cb = UsageMetadataCallbackHandler()
            if rag_mode == "Simple RAG (no memory)":
                # Simple RAG: raw question only — the limitation we demo
                chain = build_simple_chain(llm, retriever, sink)
                config = {"callbacks": [usage_cb]}
            else:
                # LCEL + memory: rewrite -> retrieve -> answer with history
                chain = build_memory_chain(llm, retriever, history_factory, sink)
                config = {
                    "configurable": {"session_id": SESSION_ID},
                    "callbacks": [usage_cb],
                }
            inputs = {"question": prompt}

            if response_mode == "invoke":
                # Wait for the full answer
                result = chain.invoke(inputs, config=config)
                answer = result.content if hasattr(result, "content") else str(result)
                if getattr(result, "usage_metadata", None):
                    meta["usage"] = result.usage_metadata
                st.markdown(answer)
            else:
                # stream / astream: write tokens as they arrive via
                # st.write_stream, which natively consumes both sync
                # iterators and async generators. For astream we bridge the
                # chain's async iterator with an `async def` generator;
                # Streamlit manages the event loop, so no asyncio.run() and
                # no blocking of the script thread is needed.
                def _sync_stream():
                    first = True
                    for chunk in chain.stream(inputs, config=config):
                        if first:
                            meta["ttft_seconds"] = time.perf_counter() - t0
                            first = False
                        if getattr(chunk, "usage_metadata", None):
                            meta["usage"] = chunk.usage_metadata
                        yield chunk.content if hasattr(chunk, "content") else str(chunk)

                async def _async_stream():
                    first = True
                    async for chunk in chain.astream(inputs, config=config):
                        if first:
                            meta["ttft_seconds"] = time.perf_counter() - t0
                            first = False
                        if getattr(chunk, "usage_metadata", None):
                            meta["usage"] = chunk.usage_metadata
                        yield chunk.content if hasattr(chunk, "content") else str(chunk)

                gen = _sync_stream() if response_mode == "stream" else _async_stream()
                answer = st.write_stream(gen).strip()

            meta["total_seconds"] = time.perf_counter() - t0
            chunks = sink.get("chunks", [])
            meta["standalone_question"] = sink.get("standalone_question")
            meta["rewrite_seconds"] = sink.get("rewrite_seconds", 0.0)
            meta["retrieve_seconds"] = sink.get("retrieve_seconds", 0.0)
            # Prefer the callback-captured usage (works for every mode);
            # fall back to usage_metadata on the result/stream chunks.
            if usage_cb.usage_metadata:
                usage = usage_cb.usage_metadata
                # Single-model runs: flatten to the standard
                # {input_tokens, output_tokens, total_tokens} shape.
                if len(usage) == 1:
                    meta["usage"] = next(iter(usage.values()))
                else:
                    meta["usage"] = usage

            render_sources(chunks)
            render_metadata(meta)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks,
                    "meta": meta,
                }
            )
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")

# Clean up uploaded temp files when the process exits
import atexit

atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))
