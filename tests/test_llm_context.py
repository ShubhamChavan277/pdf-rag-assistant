
"""Tests for LLM context formatting and token estimation."""

import pytest

from pdf_rag.llm_context import estimate_tokens, format_passages


def test_estimate_tokens_returns_zero_for_empty_text() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_uses_conservative_character_estimate() -> None:
    assert estimate_tokens("abcdef") == 2
    assert estimate_tokens("abcdefg") == 3


def test_estimate_tokens_rejects_non_string_input() -> None:
    with pytest.raises(ValueError, match="string"):
        estimate_tokens(None)


def test_format_passages_includes_source_and_page_metadata() -> None:
    passages = [
        {
            "text": "First passage content.",
            "source_name": "guide.pdf",
            "page_numbers": (2, 3),
        }
    ]

    result = format_passages(passages)

    assert "[Passage 1" in result
    assert "Source: guide.pdf" in result
    assert "Pages: 2, 3" in result
    assert "First passage content." in result


def test_format_passages_supports_single_page_number() -> None:
    result = format_passages(
        [
            {
                "text": "Passage content.",
                "source_name": "guide.pdf",
                "page_numbers": 4,
            }
        ]
    )

    assert "Pages: 4" in result


def test_format_passages_handles_missing_metadata() -> None:
    result = format_passages([{"text": "Passage content."}])

    assert "Source: Unknown source" in result
    assert "Pages: Not available" in result


def test_format_passages_formats_multiple_passages() -> None:
    result = format_passages(
        [
            {"text": "First passage.", "source_name": "a.pdf"},
            {"text": "Second passage.", "source_name": "b.pdf"},
        ]
    )

    assert "[Passage 1" in result
    assert "[Passage 2" in result
    assert result.index("First passage.") < result.index("Second passage.")


def test_format_passages_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="sequence"):
        format_passages("invalid")


def test_format_passages_rejects_invalid_passage() -> None:
    with pytest.raises(ValueError, match="mapping"):
        format_passages(["invalid"])


def test_format_passages_rejects_empty_passage_text() -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        format_passages([{"text": "  "}])
