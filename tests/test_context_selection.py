"""Tests for reranked passage context selection."""

import pytest

from pdf_rag.context_selection import select_context


def make_passages() -> list[dict]:
    """Build passages in descending relevance order."""
    return [
        {"id": "p1", "text": "First relevant passage", "score": 0.99},
        {"id": "p2", "text": "Second relevant passage", "score": 0.90},
        {"id": "p3", "text": "Third relevant passage", "score": 0.80},
        {"id": "p4", "text": "Fourth relevant passage", "score": 0.70},
    ]


def word_counter(text: str) -> int:
    """Provide a deterministic token-counting substitute."""
    return len(text.split())


def test_select_context_respects_passage_limit() -> None:
    results = select_context(
        make_passages(),
        word_counter,
        max_passages=2,
        max_context_tokens=100,
    )

    assert len(results) == 2
    assert [item["id"] for item in results] == ["p1", "p2"]


def test_select_context_respects_token_budget() -> None:
    results = select_context(
        make_passages(),
        word_counter,
        max_passages=5,
        max_context_tokens=7,
    )

    assert [item["id"] for item in results] == ["p1", "p2"]


def test_select_context_skips_passage_that_exceeds_remaining_budget() -> None:
    passages = [
        {"id": "p1", "text": "one two three", "score": 0.99},
        {"id": "p2", "text": "one two three four five", "score": 0.90},
        {"id": "p3", "text": "one two", "score": 0.80},
    ]

    results = select_context(
        passages,
        word_counter,
        max_passages=5,
        max_context_tokens=5,
    )

    assert [item["id"] for item in results] == ["p1", "p3"]


def test_select_context_preserves_passage_metadata() -> None:
    passage = {
        "id": "doc-1:0",
        "text": "Relevant passage",
        "score": 0.95,
        "document_id": "doc-1",
        "source_name": "sample.pdf",
        "chunk_index": 0,
        "page_numbers": (2, 3),
    }

    results = select_context(
        [passage],
        word_counter,
        max_context_tokens=100,
    )

    assert results[0] == passage


def test_select_context_does_not_modify_input_passages() -> None:
    passages = make_passages()
    original = [dict(item) for item in passages]

    select_context(passages, word_counter)

    assert passages == original


def test_select_context_returns_empty_for_empty_input() -> None:
    assert select_context([], word_counter) == []


@pytest.mark.parametrize(
    ("max_passages", "max_context_tokens"),
    [
        (0, 100),
        (-1, 100),
        (True, 100),
        (1.5, 100),
        (5, 0),
        (5, -1),
        (5, True),
        (5, 1.5),
    ],
)
def test_select_context_rejects_invalid_limits(
    max_passages: int | float | bool,
    max_context_tokens: int | float | bool,
) -> None:
    with pytest.raises(ValueError):
        select_context(
            make_passages(),
            word_counter,
            max_passages=max_passages,
            max_context_tokens=max_context_tokens,
        )


def test_select_context_rejects_invalid_token_counter() -> None:
    with pytest.raises(ValueError, match="callable"):
        select_context(make_passages(), None)


def test_select_context_rejects_invalid_passages() -> None:
    with pytest.raises(ValueError):
        select_context(["invalid"], word_counter)

    with pytest.raises(ValueError):
        select_context([{"text": ""}], word_counter)


def test_select_context_rejects_invalid_token_count() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        select_context(
            [{"text": "passage"}],
            lambda text: -1,
        )


def test_select_context_wraps_token_counter_errors() -> None:
    def failing_counter(text: str) -> int:
        raise RuntimeError("Tokenizer failure")

    with pytest.raises(RuntimeError, match="Failed to count"):
        select_context(
            [{"text": "passage"}],
            failing_counter,
        )
