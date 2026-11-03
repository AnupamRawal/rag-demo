"""schemas.py — Typed Pydantic schemas for the research agent.

The agent's final answer is a structured ResearchAnswer object, not free-form text.
This contract guarantees that the Streamlit UI or any downstream API client can
reliably render typed fields (answer, brief, related papers, citations, confidence,
follow-ups) instead of parsing unstructured markdown with regex.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Evidence snippet supporting a claim, strictly extracted from tool results."""

    source: str = Field(
        description="Source document name, file path, or URL from tool results."
    )
    page: int | None = Field(
        default=None,
        description="Page number if source is a PDF, otherwise null.",
    )
    quote: str = Field(
        description="Verbatim snippet (<= 25 words) from tool results supporting the claim."
    )


class PaperBrief(BaseModel):
    """Structured executive brief of a research paper.

    Populated ONLY when the user explicitly asks for a summary or brief of a document.
    """

    title: str = Field(description="Title of the paper.")
    one_line_summary: str = Field(
        description="A single clear sentence capturing the core contribution."
    )
    problem: str = Field(
        description="The fundamental problem, bottleneck, or research gap addressed."
    )
    method: str = Field(
        description="The technical architecture, model design, or algorithmic method proposed."
    )
    key_results: list[str] = Field(
        default_factory=list,
        description="Up to 5 key empirical benchmarks, scores, or headline results.",
        max_length=5,
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Up to 4 stated limitations, failure modes, or areas for future research.",
        max_length=4,
    )


class RelatedPaper(BaseModel):
    """A topically related academic paper discovered via text-based arXiv search.

    Honesty requirement: This represents topically related work discovered by text search,
    NOT a citation graph (it does not indicate papers that cite or build upon the target).
    """

    title: str = Field(description="Title of the related paper.")
    authors: list[str] = Field(
        default_factory=list,
        description="Up to 3 primary authors. If more, append 'et al.' to the last entry.",
    )
    year: int | str = Field(description="Publication year.")
    url: str = Field(description="Direct arXiv URL (e.g. https://arxiv.org/abs/...).")
    why_related: str = Field(
        description="One or two sentences explaining why this paper is topically related."
    )


class ResearchAnswer(BaseModel):
    """The complete structured response returned by the research agent."""

    answer: str = Field(
        description="Clear, comprehensive, and grounded answer to the user's question."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Self-reported confidence: 'high' if directly verified in sources, "
        "'medium' if inferred/combined, 'low' if weak or unsupported."
    )
    answered_from: list[Literal["documents", "arxiv", "none"]] = Field(
        description="Tools relied upon for this answer: ['documents'], ['arxiv'], or ['none']."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer. Every citation must come from tool results. "
        "Leave empty if answered_from is ['none'].",
    )
    brief: PaperBrief | None = Field(
        default=None,
        description="Optional paper brief. Populate ONLY when the user explicitly requests a summary "
        "or brief of a paper/document.",
    )
    related_papers: list[RelatedPaper] = Field(
        default_factory=list,
        description="List of related papers found via arxiv_search. Leave empty if arxiv_search was not used.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Up to 2 logical, high-value follow-up questions.",
        max_length=2,
    )
