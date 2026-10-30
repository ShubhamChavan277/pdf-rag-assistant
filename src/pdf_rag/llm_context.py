
"""Format selected PDF passages for grounded LLM answering."""

from collections.abc import Mapping, Sequence
from typing import Any


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens using approximately 3 characters each."""

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    if not text:
        return 0

    return (len(text) + 2) // 3


def format_passages(passages: Sequence[Mapping[str, Any]]) -> str:
    """Format selected passages with source and page attribution."""

    if isinstance(passages, (str, bytes)) or not isinstance(
        passages, Sequence
    ):
        raise ValueError("Passages must be a sequence.")

    formatted: list[str] = []

    for index, passage in enumerate(passages, start=1):
        if not isinstance(passage, Mapping):
            raise ValueError("Every passage must be a mapping.")

        text = passage.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every passage must contain non-empty text.")

        source_name = passage.get("source_name", "Unknown source")
        if not isinstance(source_name, str) or not source_name.strip():
            source_name = "Unknown source"

        page_numbers = passage.get("page_numbers", ())
        if isinstance(page_numbers, int) and not isinstance(
            page_numbers, bool
        ):
            page_numbers = (page_numbers,)

        if not isinstance(page_numbers, (list, tuple)):
            page_numbers = ()

        valid_pages = [
            str(page)
            for page in page_numbers
            if isinstance(page, int) and not isinstance(page, bool) and page > 0
        ]

        page_label = (
            ", ".join(valid_pages) if valid_pages else "Not available"
        )

        formatted.append(
            f"[Passage {index} | Source: {source_name} | "
            f"Pages: {page_label}]\n{text.strip()}"
        )

    return "\n\n".join(formatted)
