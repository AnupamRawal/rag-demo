"""agent.py — The research agent powered by langchain create_agent and LangGraph.

Core capabilities:
- Decision-making LLM agent choosing dynamically between RAG (search_documents),
  literature discovery (arxiv_search), action execution (save_brief), or direct response.
- Typed Pydantic structured output (ToolStrategy(ResearchAnswer)).
- Thread-level state persistence with InMemorySaver and JsonPlusSerializer allowlist.
- Dynamic context trimming middleware (@wrap_model_call) keeping active turn intact.
- Streaming execution (stream_mode="values") yielding real-time tool calls and results.
- LangSmith tracing integration with run URL generation.
"""

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import collect_runs
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import schemas
from schemas import Citation, PaperBrief, RelatedPaper, ResearchAnswer
from trimming import TrimmingSink, create_trimming_middleware

SYSTEM_PROMPT = """You are an expert research assistant. RAG becomes one tool among several, and you decide which tools to call.

Available Tools:
- search_documents(query, source=None): FAISS retrieval over the user's uploaded papers, text files, and web pages. Use this FIRST for questions regarding uploaded documents. Pass a concise, standalone query. Use `source` (e.g. file name or URL) to filter to a specific source when comparing or focusing on one document.
- arxiv_search(query, max_results=3, sort="relevance"|"date"): Searches arXiv for topically related research papers by text. Returns title, authors (max 3), publication year, abstract preview, and arXiv URL. Note: this is text-based topical search, NOT a citation graph.
- save_brief(filename, markdown): Action tool to save an executive paper brief as a markdown file inside briefs/. Only use this when explicitly requested by the user.

Strict Guidelines:
1. Tool Selection: If a query requires no external knowledge (e.g. greetings, pleasantries, rephrasings, simple acknowledgments), do NOT call any tool. Respond directly with answered_from=["none"].
2. Non-Empty & Specific Queries: ALWAYS pass a meaningful, specific query string to search_documents and arxiv_search. NEVER pass an empty string, whitespace-only string, or generic structural terms like "abstract", "title", "introduction", "conclusion", "LangChain", or the document filename. These terms do not retrieve useful content because the search uses semantic similarity, not keyword matching. Instead, use queries that describe the concept you want, for example: "core problem this research addresses", "proposed architecture and key components", "experimental results and performance benchmarks", "limitations and future work directions".
3. No Retry Loops on search_documents: If search_documents returns any result, do NOT call it again with a rephrased or slightly different version of the same query in the same turn. For a paper brief, make at most 2-3 targeted calls covering distinct sections (e.g. one for "problem and motivation", one for "method and architecture", one for "results and evaluation"). Accept what is returned and synthesise your answer.
3. Compound Brief + Related Papers Plan: When asked for both a paper brief AND related papers in one prompt, follow this exact plan with NO deviations — make exactly these calls in order:
   a. search_documents("core problem and motivation this paper addresses")
   b. search_documents("proposed method architecture and technical approach")
   c. search_documents("experimental results evaluation and limitations")
   d. arxiv_search("<specific topic from the paper>", max_results=3, sort="relevance")
   Then immediately write your final answer. Do NOT add extra calls.
4. No Retry Loops on search_documents: If search_documents returns any result, do NOT call it again with a rephrased or slightly different version of the same query in the same turn. Accept what is returned and synthesise your answer.
5. No Retry Loops on arxiv_search: If arxiv_search returns results (even if not ideal), do NOT call it again with a rephrased version of the same query in the same turn. Only call arxiv_search a second time if the first call targeted a genuinely different topic.
6. arXiv Query & Sort: Always use a specific, narrow topic query for arxiv_search (e.g. "retrieval-augmented generation vector databases", NOT "recent advancements in machine learning"). Broad sweep queries produce unrelated results regardless of sort. Use sort="relevance" by default — this is almost always better. Only use sort="date" when (a) the user explicitly says "most recent" AND (b) your query is already specific and narrow (e.g. "constitutional AI alignment 2024"). Never combine a broad query with sort="date".
7. Evidence Grounding: Every citation in `citations` and every paper in `related_papers` MUST originate from an actual tool result in this conversation. NEVER fabricate sources, quotes, page numbers, or arXiv links. Quote snippets must be <= 25 words copied verbatim from tool results.
8. Paper Briefs: Populate the `brief` object ONLY when the user explicitly requests an executive summary, brief, or breakdown of a paper or document. For standard Q&A questions, leave `brief` null.
9. Honest Paper Attribution: When returning related papers from arxiv_search, represent them honestly as topically related papers found by text search. Never label or claim them to be papers that cite or build upon the uploaded document.
10. Self-Reported Confidence: Set `confidence` honestly ("high" if directly supported by retrieved text, "medium" if inferred or synthesized, "low" if poorly supported or absent).
11. Missing Information: If the retrieved documents do not contain the answer, state that clearly and set confidence to "low"."""


class PerCallUsageCallbackHandler(BaseCallbackHandler):
    """Captures token usage on each LLM call to correlate with context trimming telemetry."""

    def __init__(self) -> None:
        super().__init__()
        self.call_usages: list[dict] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            gen = response.generations[0][0]
            if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
                meta = gen.message.usage_metadata or {}
                self.call_usages.append(meta)
        except Exception:
            pass


def get_checkpointer() -> InMemorySaver:
    """Create an InMemorySaver with allowlisted Pydantic classes for safe deserialization."""
    classes = [
        "ResearchAnswer",
        "Citation",
        "PaperBrief",
        "RelatedPaper",
    ]
    allowlist = []
    # Allowlist across potential import prefixes
    for mod in ["schemas", "src.post-3-research-agent.schemas", getattr(schemas, "__name__", "schemas")]:
        for cls in classes:
            allowlist.append((mod, cls))

    serde = JsonPlusSerializer(allowed_msgpack_modules=allowlist)
    return InMemorySaver(serde=serde)


def build_llm(model_name: str, temperature: float = 0.2):
    """Construct LLM — routes to ChatOpenAI for gpt-* models, ChatGroq for everything else."""
    if model_name.startswith("gpt-"):
        return ChatOpenAI(
            model=model_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=temperature,
        )
    return ChatGroq(
        model=model_name,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=temperature,
    )


def build_research_agent(
    llm: Any,
    tools: list,
    checkpointer: InMemorySaver,
    trimming_sink: TrimmingSink,
):
    """Instantiate the compiled agent with trimming middleware and structured output."""
    trimming_middleware = create_trimming_middleware(trimming_sink)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ResearchAnswer),
        checkpointer=checkpointer,
        middleware=[trimming_middleware],
    )


@dataclass
class AgentStep:
    """A live agent action or observation."""

    kind: str  # "tool_call" | "tool_result"
    name: str
    detail: str


@dataclass
class RunResult:
    """Aggregated telemetry and final answer for a single user turn."""

    structured: ResearchAnswer | None = None
    fallback_text: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    run_id: str | None = None
    steps: list[AgentStep] = field(default_factory=list)


def _preview(text: Any, limit: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"


def stream_agent(
    agent,
    question: str,
    thread_id: str,
    tags: list[str],
    metadata: dict,
    trimming_sink: TrimmingSink,
    recursion_limit: int = 12,
) -> Iterator[tuple[str, Any]]:
    """Stream agent execution: yields ('step', AgentStep) then ('done', RunResult)."""
    usage_cb = UsageMetadataCallbackHandler()
    per_call_cb = PerCallUsageCallbackHandler()

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [usage_cb, per_call_cb],
        "tags": tags,
        "metadata": metadata,
        "run_name": "rag-agent",
        "recursion_limit": recursion_limit,
    }

    result = RunResult()
    seen: int | None = None
    last_state: dict = {}
    start_time = time.perf_counter()

    with collect_runs() as collector:
        for state in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="values",
        ):
            last_state = state
            messages = state.get("messages", [])

            # On the first yielded state, capture message count from prior turns
            if seen is None:
                seen = len(messages)
                continue

            # Process only NEW messages emitted during this active turn
            for msg in messages[seen:]:
                if isinstance(msg, AIMessage):
                    result.llm_calls += 1
                    for call in getattr(msg, "tool_calls", []):
                        call_name = call.get("name", "")
                        # Exclude the internal schema tool call from displayed agent steps
                        if call_name == ResearchAnswer.__name__:
                            continue
                        result.tool_calls += 1
                        step = AgentStep("tool_call", call_name, _preview(call.get("args", {})))
                        result.steps.append(step)
                        yield "step", step

                elif isinstance(msg, ToolMessage):
                    tool_name = msg.name or "tool"
                    # Exclude internal schema tool results
                    if tool_name == ResearchAnswer.__name__:
                        continue
                    step = AgentStep("tool_result", tool_name, _preview(msg.content))
                    result.steps.append(step)
                    yield "step", step

            seen = len(messages)

        result.seconds = time.perf_counter() - start_time
        if collector.traced_runs:
            result.run_id = str(collector.traced_runs[0].id)

    # Check for structured response
    structured_resp = last_state.get("structured_response")
    if isinstance(structured_resp, ResearchAnswer):
        result.structured = structured_resp
    else:
        # Fallback to last AIMessage text content if schema was not adhered to
        for msg in reversed(last_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                result.fallback_text = str(msg.content)
                break

    # Aggregate measured token usage for this turn
    for model_usage in usage_cb.usage_metadata.values():
        result.input_tokens += model_usage.get("input_tokens", 0)
        result.output_tokens += model_usage.get("output_tokens", 0)
        result.total_tokens += model_usage.get("total_tokens", 0)

    # Correlate real input tokens with trimming telemetry records
    for i, rec in enumerate(trimming_sink.calls):
        if i < len(per_call_cb.call_usages):
            rec.real_input_tokens = per_call_cb.call_usages[i].get("input_tokens")

    yield "done", result


def get_trace_url(run_id: str | None) -> str | None:
    """Retrieve LangSmith web URL for the run. Returns None if tracing is off or fails."""
    if not run_id:
        return None
    try:
        api_key = os.environ.get("LANGSMITH_API_KEY")
        tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1")
        if not api_key or not tracing:
            return None

        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        client = Client()
        run = client.read_run(run_id)
        return client.get_run_url(run=run)
    except Exception:
        return None
