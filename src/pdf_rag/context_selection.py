"""Select reranked passages within a configurable context budget."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any


DEFAULT_MAX_CONTEXT_PASSAGES = 5
DEFAULT_CONTEXT_BUDGET_TOKENS = 2000


def select_context(
    ranked_passages: Sequence[Mapping[str, Any]],
    token_counter: Callable[[str], int],
    max_passages: int = DEFAULT_MAX_CONTEXT_PASSAGES,
    max_context_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
) -> list[dict[str, Any]]:
    """Select the highest-ranked passages within count and token limits."""
    if isinstance(ranked_passages, (str, bytes)) or not isinstance(
        ranked_passages, Sequence
    ):
        raise ValueError("Ranked passages must be a sequence.")

    if not callable(token_counter):
        raise ValueError("Token counter must be callable.")

    if (
        isinstance(max_passages, bool)
        or not isinstance(max_passages, int)
        or max_passages < 1
    ):
        raise ValueError("Maximum passages must be a positive integer.")

    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens < 1
    ):
        raise ValueError("Context token budget must be a positive integer.")

    selected: list[dict[str, Any]] = []
    total_tokens = 0

    for passage in ranked_passages:
        if not isinstance(passage, Mapping):
            raise ValueError("Every ranked passage must be a mapping.")

        text = passage.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every passage must contain non-empty text.")

        if len(selected) >= max_passages:
            break

        try:
            token_count = token_counter(text)
        except Exception as exc:
            raise RuntimeError("Failed to count passage tokens.") from exc

        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("Token counter must return a non-negative integer.")

        if total_tokens + token_count > max_context_tokens:
            continue

        selected.append(dict(passage))
        total_tokens += token_count

    return selected
