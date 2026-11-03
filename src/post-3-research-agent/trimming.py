"""trimming.py — Context trimming middleware for the research agent.

Implements an AgentMiddleware via @wrap_model_call to intercept model requests.
The full conversation history remains intact in the LangGraph checkpointer, while
the middleware dynamically trims what the LLM *sees* on each call.

Trimming Rules:
1. Always keep the current turn (the latest user message and all subsequent
   tool calls / results in the active turn) whole and untrimmed.
2. Older turns are trimmed newest-first using `trim_messages`:
   - token_counter="approximate" (~4 chars per token)
   - strategy="last"
   - start_on="human" (never leave an orphan AI or tool message)
   - allow_partial=False (preserve full message blocks)
3. Collects telemetry per model call:
   - messages in memory vs messages sent
   - approximate tokens in memory vs approximate tokens sent
   - real input tokens (correlated from UsageMetadataCallbackHandler)
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately


@dataclass
class ModelCallTrimmingRecord:
    """Measured trimming statistics for a single LLM invocation."""

    call_index: int
    messages_in_memory: int
    messages_sent: int
    approx_tokens_in_memory: int
    approx_tokens_sent: int
    real_input_tokens: int | None = None


@dataclass
class TrimmingSink:
    """Per-turn collector for trimming telemetry and user configuration."""

    calls: list[ModelCallTrimmingRecord] = field(default_factory=list)
    trimming_enabled: bool = True
    max_tokens: int = 4000

    def reset(self) -> None:
        self.calls.clear()


def apply_context_trimming(
    messages: list[BaseMessage],
    max_tokens: int = 4000,
    trimming_enabled: bool = True,
) -> tuple[list[BaseMessage], int, int]:
    """Trim older conversation turns while keeping the current turn intact.

    Returns:
        (trimmed_messages, approx_tokens_in_memory, approx_tokens_sent)
    """
    approx_in_memory = count_tokens_approximately(messages)

    if not trimming_enabled or not messages:
        return list(messages), approx_in_memory, approx_in_memory

    # Find where the current turn starts (the latest HumanMessage)
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human_idx = i
            break

    # If no HumanMessage found, treat entire sequence as current turn
    if last_human_idx == -1:
        return list(messages), approx_in_memory, approx_in_memory

    older_messages = messages[:last_human_idx]
    current_turn = messages[last_human_idx:]

    # Always keep current turn whole
    current_turn_tokens = count_tokens_approximately(current_turn)
    remaining_budget = max(0, max_tokens - current_turn_tokens)

    if older_messages and remaining_budget > 0:
        trimmed_older = trim_messages(
            older_messages,
            max_tokens=remaining_budget,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
    else:
        trimmed_older = []

    final_messages = list(trimmed_older) + list(current_turn)
    approx_sent = count_tokens_approximately(final_messages)

    return final_messages, approx_in_memory, approx_sent


def create_trimming_middleware(sink: TrimmingSink) -> AgentMiddleware:
    """Construct an AgentMiddleware instance that trims context and records telemetry."""

    @wrap_model_call
    def trimming_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        final_messages, approx_mem, approx_sent = apply_context_trimming(
            messages=request.messages,
            max_tokens=sink.max_tokens,
            trimming_enabled=sink.trimming_enabled,
        )

        record = ModelCallTrimmingRecord(
            call_index=len(sink.calls) + 1,
            messages_in_memory=len(request.messages),
            messages_sent=len(final_messages),
            approx_tokens_in_memory=approx_mem,
            approx_tokens_sent=approx_sent,
            real_input_tokens=None,  # Populated after run via usage handler
        )
        sink.calls.append(record)

        new_request = request.override(messages=final_messages)
        return handler(new_request)

    return trimming_middleware
