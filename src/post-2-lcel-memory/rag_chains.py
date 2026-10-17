"""
rag_chains.py — the two RAG pipelines, built with LCEL.

Simple RAG (no memory):
    retriever -> prompt -> llm -> StrOutputParser
    The retriever only ever sees the raw user question.

LCEL + memory:
    Step 1: question-rewrite chain  (history + latest question -> standalone question)
    Step 2: retrieve with the standalone question
    Step 3: answer prompt (system rules, MessagesPlaceholder("history"), context,
            question) -> llm, wrapped in RunnableWithMessageHistory keyed by
            session_id with an in-memory ChatMessageHistory store.
"""

import time

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_groq import ChatGroq

# Prompts
SIMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY from the provided "
            "document context. If the answer is not in the context, say "
            "\"That is not in the document.\"\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and a follow-up question, rewrite the "
            "follow-up as a standalone question that can be understood "
            "without the chat history. Do NOT answer it — only output the "
            "rewritten question.",
        ),
        ("human", "Chat history:\n{history}\n\nFollow-up question: {question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY from the provided "
            "document context. If the answer is not in the context, say "
            "\"That is not in the document.\"\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def format_history(messages) -> str:
    """Flatten chat messages to plain text for the rewrite prompt."""
    return "\n".join(f"{m.type}: {m.content}" for m in messages[-8:])


def make_llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


def build_simple_chain(llm, retriever, metadata_sink: dict | None = None):
    """
    LCEL: retriever -> prompt -> llm -> StrOutputParser.

    RunnablePassthrough keeps the raw question flowing through; the context
    is fetched with that same (unmodified) question — this is exactly the
    limitation we demonstrate: a follow-up like "why is it better?" is used
    as-is for retrieval and misses the real subject ("multi-head attention").

    metadata_sink (optional) records the retrieved chunks so the sources
    panel can show them.
    """

    def retrieve_step(x):
        t0 = time.perf_counter()
        docs = retriever.invoke(x["question"])
        if metadata_sink is not None:
            metadata_sink["chunks"] = docs
            metadata_sink["retrieve_seconds"] = time.perf_counter() - t0
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return (
        RunnablePassthrough.assign(context=retrieve_step)
        | SIMPLE_PROMPT
        | llm
        | StrOutputParser()
    )


def build_memory_chain(llm, retriever, get_session_history, metadata_sink: dict):
    """
    Full conversational RAG, built entirely with LCEL and wrapped in
    RunnableWithMessageHistory.

    metadata_sink is a plain dict the app reads after the run to show the
    standalone question and per-step latencies in the metadata panel.
    """

    # Skips the rewrite LLM call when history is empty
    def rewrite_step(inputs):
        history = inputs.get("history") or []
        question = inputs["question"]
        if not history:  # empty history -> nothing to resolve
            metadata_sink["standalone_question"] = question
            metadata_sink["rewrite_seconds"] = 0.0
            return question
        t0 = time.perf_counter()
        rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
        standalone = rewrite_chain.invoke(
            {"question": question, "history": format_history(history)}
        ).strip()
        metadata_sink["standalone_question"] = standalone
        metadata_sink["rewrite_seconds"] = time.perf_counter() - t0
        return standalone

    # Retrieve with the standalone (rewritten) question
    def retrieve_step(inputs):
        t0 = time.perf_counter()
        docs = retriever.invoke(inputs["standalone_question"])
        metadata_sink["retrieve_seconds"] = time.perf_counter() - t0
        metadata_sink["chunks"] = docs
        return "\n\n---\n\n".join(d.page_content for d in docs)

    full_chain = (
        RunnablePassthrough.assign(standalone_question=rewrite_step).assign(
            context=retrieve_step
        )
        | ANSWER_PROMPT
        | llm
    )

    return RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def get_history_factory(store: dict):
    """Return a get_session_history function backed by an in-memory dict."""

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return get_session_history

