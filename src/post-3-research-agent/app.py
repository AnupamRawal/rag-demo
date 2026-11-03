"""app.py — Streamlit UI for Post 3: Agentic RAG, Tools & Structured API Contract.

Key Highlights:
- "RAG becomes one tool among several, and the agent decides."
- Multi-source ingestion: PDF, TXT/MD, and Web URLs with fingerprint index caching.
- Tool selection: search_documents (with source filtering), arxiv_search (text-based related papers),
  and save_brief (disk persistence with safety checks).
- Context trimming middleware keeping current turn intact while bounding token costs.
- Fully measured metadata panel: latency, tokens, tool durations, trimming metrics.
- Pydantic structured output rendered as a card with paper brief and clickable follow-ups.
- LangSmith tracing integration.
"""

import atexit
import json
import os
import shutil
import tempfile
import time
import uuid

# Environment configuration must occur before transformers/torch imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "rag-demo-post-3/1.0"

import streamlit as st
from dotenv import load_dotenv

from agent import (
    RunResult,
    build_llm,
    build_research_agent,
    get_checkpointer,
    get_trace_url,
    stream_agent,
)
from loaders import (
    SourceSpec,
    build_vector_store,
    chunk_documents,
    get_embeddings,
    hash_bytes,
    load_source,
)
from schemas import ResearchAnswer
from tools import ToolSink, make_tools
from trimming import TrimmingSink

load_dotenv()

st.set_page_config(
    page_title="Post 3: Research Agent (Agentic RAG)",
    page_icon="🔬",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Temp directory management
# ------------------------------------------------------------------------------
if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp(prefix="research_agent_")

TMP_DIR = st.session_state.tmp_dir
atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))

RECURSION_LIMIT = 50  # max LangGraph steps per turn; GPT-4o uses more steps per tool call


def save_upload(uploaded_file) -> tuple[str, str]:
    """Persist an UploadedFile to temp directory and return (path, content_hash)."""
    file_bytes = uploaded_file.getvalue()
    content_hash = hash_bytes(file_bytes)
    path = os.path.join(TMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path, content_hash


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_checkpointer()

if "tool_sink" not in st.session_state:
    st.session_state.tool_sink = ToolSink()

if "trimming_sink" not in st.session_state:
    st.session_state.trimming_sink = TrimmingSink()

# ------------------------------------------------------------------------------
# Sidebar Layout
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Agent")
    st.caption("Post 3: Tools, Structured Output & Context Trimming")

    # Section 1: Document sources
    st.header("1. Document sources")
    pdf_file = st.file_uploader("PDF Paper (PyPDFLoader)", type=["pdf"])
    text_file = st.file_uploader("Text / Markdown (TextLoader)", type=["txt", "md"])
    url_input = st.text_input("Web URL (WebBaseLoader)", placeholder="https://...")

    specs: list[SourceSpec] = []
    if pdf_file:
        pdf_path, pdf_hash = save_upload(pdf_file)
        specs.append(
            SourceSpec(
                source_type="pdf",
                loader_name="PyPDFLoader",
                path_or_url=pdf_path,
                display_name=pdf_file.name,
                content_hash=pdf_hash,
            )
        )
    if text_file:
        txt_path, txt_hash = save_upload(text_file)
        specs.append(
            SourceSpec(
                source_type="text",
                loader_name="TextLoader",
                path_or_url=txt_path,
                display_name=text_file.name,
                content_hash=txt_hash,
            )
        )
    if url_input.strip():
        clean_url = url_input.strip()
        specs.append(
            SourceSpec(
                source_type="url",
                loader_name="WebBaseLoader",
                path_or_url=clean_url,
                display_name=clean_url,
                content_hash=clean_url,
            )
        )

    st.divider()

    # Section 2: Tools
    st.header("2. Tools")
    use_docs = st.checkbox("search_documents", value=True, help="FAISS retrieval over uploaded sources")
    use_arxiv = st.checkbox("arxiv_search", value=True, help="Live arXiv search for topically related literature")
    use_save = st.checkbox(
        "save_brief",
        value=False,
        help="Action tool: safely saves executive briefs to src/post-3-research-agent/briefs/",
    )

    st.divider()

    # Section 3: Model
    st.header("3. Model")
    groq_model = st.selectbox(
        "Model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ],
        index=0,
        help="gpt-4o / gpt-4o-mini require OPENAI_API_KEY in .env. Groq models require GROQ_API_KEY.",
    )
    if groq_model.startswith("gpt-") and not os.environ.get("OPENAI_API_KEY"):
        st.warning("⚠️ Add OPENAI_API_KEY to your .env to use OpenAI models.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Top K chunks", 1, 10, 4)

    st.divider()

    # Section 4: Context
    st.header("4. Context")
    trim_enabled = st.checkbox("Context trimming", value=True)
    trim_budget = st.slider(
        "Context budget (tokens)",
        500,
        12000,
        4000,
        100,
        help="Budget is approximate (~4 chars/token) and excludes the system prompt and tool definitions. "
        "Trimming always keeps the current turn whole and trims older history newest-first.",
    )

    st.divider()

    # LangSmith Tracing Status
    has_langsmith_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    tracing_env = os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1")
    if has_langsmith_key and tracing_env:
        st.caption("🟢 **LangSmith tracing:** active")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        st.caption("⚪ **LangSmith tracing:** off (no API key or disabled)")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        st.session_state.tool_sink.reset()
        st.session_state.trimming_sink.reset()
        st.rerun()

    # Loaded documents summary
    if "summaries" in st.session_state and st.session_state.summaries:
        st.divider()
        st.subheader("📄 Loaded documents")
        for s in st.session_state.summaries:
            if s["error"]:
                st.error(f"❌ {s['source']} ({s['loader']}): {s['error']}")
            else:
                st.markdown(
                    f"**{s['source']}** · `{s['loader']}`\n\n"
                    f"{s['docs']} docs → {s['chunks']} chunks ({s['load_s']}s)"
                )
        if "index_seconds" in st.session_state:
            st.caption(f"⚡ Index built in {st.session_state.index_seconds}s")

# ------------------------------------------------------------------------------
# Document Indexing (Fingerprint Caching)
# ------------------------------------------------------------------------------
fingerprint = tuple((s.source_type, s.display_name, s.content_hash) for s in specs)

if specs and (
    st.session_state.get("index_fingerprint") != fingerprint
    or "vectorstore" not in st.session_state
):
    with st.sidebar, st.spinner("Loading and indexing documents..."):
        all_chunks = []
        summaries = []
        for spec in specs:
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

            chunks = chunk_documents(res.docs, chunk_size=1000, chunk_overlap=200)
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

        if all_chunks:
            t0 = time.perf_counter()
            st.session_state.vectorstore = build_vector_store(all_chunks, get_embeddings())
            st.session_state.index_seconds = round(time.perf_counter() - t0, 3)
            st.session_state.index_fingerprint = fingerprint
            st.session_state.summaries = summaries
        else:
            st.session_state.vectorstore = None
            st.session_state.summaries = summaries
            st.error("No valid text could be indexed from provided sources.")

elif not specs:
    st.session_state.vectorstore = None
    st.session_state.summaries = []
    st.session_state.index_fingerprint = None

vectorstore = st.session_state.get("vectorstore")

# ------------------------------------------------------------------------------
# Tool & Agent Assembly
# ------------------------------------------------------------------------------
enabled_tools = set()
if use_docs and vectorstore is not None:
    enabled_tools.add("search_documents")
if use_arxiv:
    enabled_tools.add("arxiv_search")
if use_save:
    enabled_tools.add("save_brief")

tool_sink: ToolSink = st.session_state.tool_sink
trimming_sink: TrimmingSink = st.session_state.trimming_sink
trimming_sink.trimming_enabled = trim_enabled
trimming_sink.max_tokens = trim_budget

tools = make_tools(
    enabled_tool_names=enabled_tools,
    sink=tool_sink,
    vectorstore=vectorstore,
    top_k=top_k,
)
tool_names = [t.name for t in tools]

# Build LLM and Agent
try:
    llm = build_llm(groq_model, temperature=temperature)
    agent = build_research_agent(
        llm=llm,
        tools=tools,
        checkpointer=st.session_state.checkpointer,
        trimming_sink=trimming_sink,
    )
except Exception as e:
    st.error(f"Error configuring Groq LLM: {type(e).__name__}: {e}")
    st.stop()


# ------------------------------------------------------------------------------
# UI Helpers & Renderers
# ------------------------------------------------------------------------------
CONFIDENCE_COLORS = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}


def queue_question(q: str) -> None:
    """Store question to be executed on the next rerun cycle."""
    st.session_state.pending_prompt = q


def render_response_card(msg: dict, is_last: bool) -> None:
    """Render structured ResearchAnswer card or fallback text."""
    result: RunResult = msg["result"]
    answer_obj = result.structured

    if answer_obj is None:
        st.warning("⚠️ Model returned raw unparsed text instead of the structured schema:")
        st.markdown(result.fallback_text or "*(No response generated)*")
    else:
        # Main answer text
        st.markdown(answer_obj.answer)

        # Paper brief (if populated)
        if answer_obj.brief:
            with st.container(border=True):
                st.markdown(f"#### 📑 Executive Brief: {answer_obj.brief.title}")
                st.markdown(f"**One-Line Summary:** *{answer_obj.brief.one_line_summary}*")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Core Problem:**")
                    st.markdown(answer_obj.brief.problem)
                with col2:
                    st.markdown("**Proposed Method:**")
                    st.markdown(answer_obj.brief.method)

                if answer_obj.brief.key_results:
                    st.markdown("**Key Empirical Results:**")
                    for r in answer_obj.brief.key_results:
                        st.markdown(f"- {r}")

                if answer_obj.brief.limitations:
                    st.markdown("**Limitations & Future Work:**")
                    for lim in answer_obj.brief.limitations:
                        st.markdown(f"- {lim}")

        # Topically Related Papers (arXiv)
        if answer_obj.related_papers:
            st.markdown("#### 📚 Related Papers (arXiv Text Search)")
            st.caption("Topically related papers discovered by text search; not a citation graph.")
            for paper in answer_obj.related_papers:
                authors = ", ".join(paper.authors) if paper.authors else "Unknown"
                st.markdown(
                    f"- **[{paper.title}]({paper.url})** ({paper.year}) — *{authors}*\n"
                    f"  *{paper.why_related}*"
                )

        # Citations & Evidence
        if answer_obj.citations:
            st.markdown("#### 🔍 Citations & Evidence")
            for c in answer_obj.citations:
                page_info = f", page {c.page}" if c.page is not None else ""
                st.markdown(f"- **`{c.source}`{page_info}**: *“{c.quote}”*")

        # Confidence Badge & Source Attribution
        color = CONFIDENCE_COLORS.get(answer_obj.confidence, "gray")
        st.markdown(
            f"**Confidence:** :{color}[**{answer_obj.confidence.upper()}**] *(self-reported by the model)* · "
            f"**Answered from:** `{', '.join(answer_obj.answered_from)}`"
        )

        # Follow-up buttons on the latest answer only
        if is_last and answer_obj.follow_up_questions:
            st.caption("Suggested follow-ups:")
            cols = st.columns(len(answer_obj.follow_up_questions))
            for i, q in enumerate(answer_obj.follow_up_questions):
                with cols[i]:
                    st.button(
                        f"↪ {q}",
                        key=f"btn_fu_{msg['id']}_{i}",
                        on_click=queue_question,
                        args=(q,),
                        use_container_width=True,
                    )

        # Raw Structured JSON expander
        with st.expander("📦 Raw Structured JSON (what the UI renders)"):
            st.caption("Structured JSON contract consumed by frontend or API clients:")
            st.json(answer_obj.model_dump())

    # Step-by-step reasoning expander
    if msg.get("steps"):
        with st.expander(f"🛠 Agent steps ({result.tool_calls} tool calls)"):
            for step in msg["steps"]:
                if step.kind == "tool_call":
                    st.markdown(f"🔧 **Tool Call:** `{step.name}`({step.detail})")
                else:
                    st.caption(f"↩︎ **Tool Result:** `{step.name}`: {step.detail}")

    # Retrieved Chunks expander
    if msg.get("chunks"):
        with st.expander(f"🔎 Retrieved chunks ({len(msg['chunks'])})"):
            for idx, doc in enumerate(msg["chunks"], 1):
                page = doc.metadata.get("page")
                page_str = f", page {page + 1}" if isinstance(page, int) else ""
                loader_name = doc.metadata.get("loader", "Loader")
                src_name = doc.metadata.get("source_name") or doc.metadata.get("source", "doc")
                st.markdown(f"**{idx}.** `{loader_name}` · `{src_name}`{page_str}")
                st.text(doc.page_content[:500])

    # Metadata expander (all measured at runtime)
    with st.expander("📊 Measured Metadata"):
        meta = msg["meta"]
        st.markdown(
            f"Model: **{meta['model']}** · Temp: **{meta['temperature']}** · "
            f"Top-K: **{meta['top_k']}** · Active tools: `{', '.join(meta['tools_enabled'])}`"
        )
        st.markdown(
            f"- Total Latency: **{meta['total_latency_seconds']:.3f}s** · "
            f"LLM Calls: **{meta['llm_calls']}** · Tool Calls: **{meta['tool_calls']}** · "
            f"Total Tool Time: **{meta['total_tool_seconds']:.3f}s**"
        )
        st.markdown(f"- Real Token Usage: `{json.dumps(meta['tokens'])}`")

        if meta.get("tool_details"):
            st.markdown("**Per-Tool Telemetry:**")
            st.dataframe(meta["tool_details"], use_container_width=True)

        if meta.get("trimming"):
            t_info = meta["trimming"]
            st.markdown(
                f"**Context Trimming:** {'Enabled' if t_info['enabled'] else 'Disabled'} "
                f"(budget: {t_info['budget_tokens']} tokens)"
            )
            if t_info.get("calls"):
                for call_stat in t_info["calls"]:
                    real_in = f"{call_stat['real_input_tokens']} real input tokens" if call_stat.get("real_input_tokens") is not None else "token count pending"
                    st.caption(
                        f"• **LLM Call {call_stat['call_index']}**: "
                        f"{call_stat['messages_in_memory']} msgs in memory ({call_stat['approx_tokens_in_memory']} approx tokens) → "
                        f"{call_stat['messages_sent']} msgs sent ({call_stat['approx_tokens_sent']} approx tokens) · "
                        f"[{real_in}]"
                    )

        if msg.get("trace_url"):
            st.link_button("🔗 Open this run in LangSmith", msg["trace_url"])
        elif meta.get("run_id"):
            st.caption(f"LangSmith Run ID: `{meta['run_id']}`")


# ------------------------------------------------------------------------------
# Main Page Chat Render
# ------------------------------------------------------------------------------
st.title("🔬 Paper Research Assistant")
st.caption(
    "Post 3: 'RAG becomes one tool among several, and the agent decides.' · "
    f"Active tools: `{', '.join(tool_names) or 'None'}` · Thread: `{st.session_state.thread_id}`"
)

if not vectorstore and use_docs:
    st.info("👈 Upload a PDF, text file, or enter a URL in the sidebar to enable document retrieval.")

# Render prior chat history
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_response_card(message, is_last=(index == len(st.session_state.messages) - 1))

# Prompt input (chat input box or queued follow-up button)
input_prompt = st.chat_input("Ask a question about your documents, research papers, or arXiv…")
pending = st.session_state.pop("pending_prompt", None)
user_query = pending or input_prompt

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    tool_sink.reset()
    trimming_sink.reset()

    meta_tags = ["post-3", groq_model, *[f"tool:{t}" for t in tool_names]]
    meta_extra = {
        "temperature": temperature,
        "top_k": top_k,
        "tools": tool_names,
        "trimming_enabled": trim_enabled,
        "trim_budget": trim_budget,
        "index_fingerprint": str(fingerprint),
    }

    with st.chat_message("assistant"):
        run_res: RunResult | None = None
        with st.status("Agent reasoning & selecting tools…", expanded=True) as status:
            try:
                for kind, payload in stream_agent(
                    agent=agent,
                    question=user_query,
                    thread_id=st.session_state.thread_id,
                    tags=meta_tags,
                    metadata=meta_extra,
                    trimming_sink=trimming_sink,
                    recursion_limit=RECURSION_LIMIT,
                ):
                    if kind == "step":
                        if payload.kind == "tool_call":
                            st.markdown(f"🔧 **Tool Call:** `{payload.name}`({payload.detail})")
                        else:
                            st.caption(f"↩︎ **Tool Result:** `{payload.name}`: {payload.detail}")
                    elif kind == "done":
                        run_res = payload

                status.update(label="Complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Execution Error: {type(e).__name__}: {e}")

        if run_res:
            trace_link = get_trace_url(run_res.run_id)

            total_tool_time = sum(c.seconds for c in tool_sink.calls)
            tool_details = [
                {
                    "tool": c.name,
                    "args": str(c.args),
                    "duration_seconds": round(c.seconds, 3),
                    "output_chars": c.output_chars,
                    "error": c.error,
                }
                for c in tool_sink.calls
            ]

            trimming_telemetry = {
                "enabled": trim_enabled,
                "budget_tokens": trim_budget,
                "calls": [
                    {
                        "call_index": r.call_index,
                        "messages_in_memory": r.messages_in_memory,
                        "messages_sent": r.messages_sent,
                        "approx_tokens_in_memory": r.approx_tokens_in_memory,
                        "approx_tokens_sent": r.approx_tokens_sent,
                        "real_input_tokens": r.real_input_tokens,
                    }
                    for r in trimming_sink.calls
                ],
            }

            measured_meta = {
                "model": groq_model,
                "temperature": temperature,
                "top_k": top_k,
                "tools_enabled": tool_names,
                "total_latency_seconds": round(run_res.seconds, 3),
                "llm_calls": run_res.llm_calls,
                "tool_calls": run_res.tool_calls,
                "total_tool_seconds": round(total_tool_time, 3),
                "tokens": {
                    "input": run_res.input_tokens,
                    "output": run_res.output_tokens,
                    "total": run_res.total_tokens,
                },
                "tool_details": tool_details,
                "trimming": trimming_telemetry,
                "run_id": run_res.run_id,
            }

            assistant_msg = {
                "role": "assistant",
                "id": uuid.uuid4().hex[:6],
                "result": run_res,
                "steps": list(run_res.steps),
                "chunks": list(tool_sink.chunks),
                "meta": measured_meta,
                "trace_url": trace_link,
            }
            st.session_state.messages.append(assistant_msg)
            st.rerun()
