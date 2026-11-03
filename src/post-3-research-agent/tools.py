"""tools.py — Specialized tools for the research agent.

Available tools:
1. search_documents(query, source=None): FAISS retrieval over uploaded papers and web pages.
   Supports metadata source filtering with fetch_k=50.
2. arxiv_search(query, max_results=3, sort='relevance'|'date'): Text-based search on arXiv.
   Finds topically related papers (honestly documented as text search, not a citation graph).
3. save_brief(filename, markdown): Action tool (default OFF in sidebar) to safely persist
   executive paper briefs inside src/post-3-research-agent/briefs/.

All tools log execution telemetry (arguments, duration, output size, errors) to a ToolSink
for the measured metadata panel. Tools catch exceptions and return informative error strings
so the agent can reason about failures and self-recover.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.tools import tool


@dataclass
class ToolCallRecord:
    """Measured telemetry for a single tool execution."""

    name: str
    args: dict[str, Any]
    seconds: float
    output_chars: int
    error: bool = False


@dataclass
class ToolSink:
    """Per-question telemetry collector for tool executions and retrieved chunks."""

    calls: list[ToolCallRecord] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)

    def reset(self) -> None:
        self.calls.clear()
        self.chunks.clear()


def slugify_filename(filename: str) -> str:
    """Sanitize filename to a safe basename, ensuring .md extension."""
    base = os.path.basename(filename).strip()
    if not base:
        base = "paper_brief.md"

    # Remove existing extension
    stem, ext = os.path.splitext(base)
    # Slugify stem: alphanumeric, hyphens, and underscores only
    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "paper_brief"

    return f"{safe_stem}.md"


def make_tools(
    enabled_tool_names: set[str],
    sink: ToolSink,
    vectorstore=None,
    top_k: int = 4,
    briefs_dir: str | None = None,
    max_calls_per_turn: int = 5,
) -> list:
    """Build and return only the tools enabled by the user in the sidebar.

    max_calls_per_turn: hard cap on total tool invocations per turn across all
    tools. Any call beyond the cap is immediately rejected with an error string
    so the agent is forced to synthesise from what it already retrieved.
    """
    tools: list = []

    if briefs_dir is None:
        briefs_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "briefs")
        )

    def _record(name: str, args: dict, start_time: float, output: str, error: bool = False) -> None:
        sink.calls.append(
            ToolCallRecord(
                name=name,
                args=args,
                seconds=time.perf_counter() - start_time,
                output_chars=len(output),
                error=error,
            )
        )

    def _check_limit(name: str, args: dict, start_time: float) -> str | None:
        """Return an error string and record it if the per-turn cap is exceeded."""
        if max_calls_per_turn > 0 and len(sink.calls) >= max_calls_per_turn:
            msg = (
                f"STOP. Tool call limit of {max_calls_per_turn} per turn has been reached. "
                "You MUST NOT call any more tools. "
                "Immediately write your final structured answer using ONLY the information already retrieved. "
                "Do not attempt to call any tool again."
            )
            _record(name, args, start_time, msg, error=True)
            return msg
        return None

    # --------------------------------------------------------------------------
    # Tool 1: search_documents
    # --------------------------------------------------------------------------
    if "search_documents" in enabled_tool_names and vectorstore is not None:

        @tool
        def search_documents(query: str, source: str | None = None) -> str:
            """Search the user's uploaded files and web pages. Use this FIRST for any question \
about the uploaded content. Pass a short, standalone search query. Optionally pass `source` \
(e.g. file name or URL) to filter results to a specific document."""
            t0 = time.perf_counter()
            try:
                if not query or not query.strip():
                    err_msg = "search_documents error: query must be a non-empty string. Provide a specific search query."
                    _record("search_documents", {"query": query, "source": source}, t0, err_msg, error=True)
                    return err_msg

                if limit_msg := _check_limit("search_documents", {"query": query, "source": source}, t0):
                    return limit_msg
                if source and source.strip():
                    src_clean = source.strip().lower()

                    def source_filter(metadata: dict) -> bool:
                        meta_src = str(
                            metadata.get("source_name")
                            or metadata.get("source")
                            or ""
                        ).lower()
                        return src_clean in meta_src

                    # FAISS post-filters after candidate retrieval, so fetch_k must be large
                    fetch_k = max(50, top_k * 10)
                    docs = vectorstore.similarity_search(
                        query,
                        k=top_k,
                        filter=source_filter,
                        fetch_k=fetch_k,
                    )
                else:
                    docs = vectorstore.similarity_search(query, k=top_k)

                sink.chunks.extend(docs)
                if not docs:
                    msg = (
                        f"No relevant passages found for query: '{query}'"
                        + (f" with source filter: '{source}'." if source else ".")
                    )
                    _record("search_documents", {"query": query, "source": source}, t0, msg)
                    return msg

                parts = []
                for i, doc in enumerate(docs, 1):
                    page = doc.metadata.get("page")
                    source_label = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
                    page_str = f", page {page + 1}" if isinstance(page, int) else ""
                    parts.append(
                        f"[{i}] (Source: {source_label}{page_str})\n{doc.page_content}"
                    )
                output = "\n\n---\n\n".join(parts)
                _record("search_documents", {"query": query, "source": source}, t0, output)
                return output

            except Exception as e:
                err_msg = f"search_documents error: {type(e).__name__}: {e}"
                _record("search_documents", {"query": query, "source": source}, t0, err_msg, error=True)
                return err_msg

        tools.append(search_documents)

    # --------------------------------------------------------------------------
    # Tool 2: arxiv_search
    # --------------------------------------------------------------------------
    if "arxiv_search" in enabled_tool_names:

        @tool
        def arxiv_search(
            query: str,
            max_results: int = 3,
            sort: Literal["relevance", "date"] = "relevance",
        ) -> str:
            """Search arXiv for topically related research papers by text. Returns title, \
authors (max 3), publication year, abstract preview, and arXiv URL. \
Note: This performs text search for topically related papers; it is NOT a citation graph."""
            t0 = time.perf_counter()
            try:
                if not query or not query.strip():
                    err_msg = "arxiv_search error: query must be a non-empty string. Provide a specific search query."
                    _record("arxiv_search", {"query": query, "max_results": max_results, "sort": sort}, t0, err_msg, error=True)
                    return err_msg

                if limit_msg := _check_limit("arxiv_search", {"query": query, "max_results": max_results, "sort": sort}, t0):
                    return limit_msg
                import arxiv

                criterion = (
                    arxiv.SortCriterion.Relevance
                    if sort == "relevance"
                    else arxiv.SortCriterion.SubmittedDate
                )
                safe_max = min(max(1, max_results), 5)
                client = arxiv.Client(
                    page_size=safe_max,
                    delay_seconds=1.0,
                    num_retries=2,
                )
                search = arxiv.Search(
                    query=query,
                    max_results=safe_max,
                    sort_by=criterion,
                )

                results = list(client.results(search))
                if not results:
                    msg = f"No arXiv papers found matching query: '{query}'."
                    _record("arxiv_search", {"query": query, "max_results": max_results, "sort": sort}, t0, msg)
                    return msg

                formatted_papers = []
                for idx, paper in enumerate(results, 1):
                    authors = [a.name for a in paper.authors]
                    if len(authors) > 3:
                        author_str = ", ".join(authors[:3]) + " et al."
                    else:
                        author_str = ", ".join(authors) if authors else "Unknown"

                    year = paper.published.year if paper.published else "N/A"
                    abstract = " ".join(paper.summary.split())
                    if len(abstract) > 380:
                        abstract = abstract[:380] + "…"

                    formatted_papers.append(
                        f"[{idx}] Title: {paper.title}\n"
                        f"Authors: {author_str}\n"
                        f"Year: {year}\n"
                        f"URL: {paper.entry_id}\n"
                        f"Abstract: {abstract}"
                    )

                output = "\n\n---\n\n".join(formatted_papers)
                _record("arxiv_search", {"query": query, "max_results": max_results, "sort": sort}, t0, output)
                return output

            except Exception as e:
                err_msg = f"arxiv_search error: {type(e).__name__}: {e}"
                _record("arxiv_search", {"query": query, "max_results": max_results, "sort": sort}, t0, err_msg, error=True)
                return err_msg

        tools.append(arxiv_search)

    # --------------------------------------------------------------------------
    # Tool 3: save_brief
    # --------------------------------------------------------------------------
    if "save_brief" in enabled_tool_names:

        @tool
        def save_brief(filename: str, markdown: str) -> str:
            """Save an executive paper brief or summary as a markdown file inside the briefs/ \
directory. Filename is automatically slugified and capped to 20 KB. Never overwrites existing files."""
            t0 = time.perf_counter()
            try:
                # Enforce directory boundary
                os.makedirs(briefs_dir, exist_ok=True)
                real_briefs_dir = os.path.realpath(briefs_dir)

                # Cap size: 20 KB limit
                raw_bytes = markdown.encode("utf-8")
                if len(raw_bytes) > 20 * 1024:
                    err_msg = (
                        f"save_brief error: File size ({len(raw_bytes)} bytes) "
                        "exceeds the 20 KB safety cap."
                    )
                    _record("save_brief", {"filename": filename}, t0, err_msg, error=True)
                    return err_msg

                # Slugify and sanitize path
                safe_name = slugify_filename(filename)
                target_path = os.path.realpath(os.path.join(real_briefs_dir, safe_name))

                # Verify target stays inside briefs directory (prevent directory traversal)
                if not (
                    target_path == real_briefs_dir
                    or target_path.startswith(real_briefs_dir + os.sep)
                ):
                    err_msg = "save_brief error: Path traversal attempt detected."
                    _record("save_brief", {"filename": filename}, t0, err_msg, error=True)
                    return err_msg

                # Never overwrite: assign numeric suffix if exists
                if os.path.exists(target_path):
                    stem, ext = os.path.splitext(safe_name)
                    counter = 1
                    while True:
                        candidate_name = f"{stem}_{counter}{ext}"
                        candidate_path = os.path.join(real_briefs_dir, candidate_name)
                        if not os.path.exists(candidate_path):
                            target_path = candidate_path
                            break
                        counter += 1

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(markdown)

                success_msg = f"Brief successfully saved to: {target_path}"
                _record("save_brief", {"filename": filename, "saved_path": target_path}, t0, success_msg)
                return success_msg

            except Exception as e:
                err_msg = f"save_brief error: {type(e).__name__}: {e}"
                _record("save_brief", {"filename": filename}, t0, err_msg, error=True)
                return err_msg

        tools.append(save_brief)

    return tools
