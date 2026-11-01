"""Tests for FlashRank passage reranking."""

from collections.abc import Sequence
import sys
from types import ModuleType
from typing import Any

import pytest

from pdf_rag.reranking import PassageReranker


class FakeRanker:
    """Simulate FlashRank without downloading or loading a model."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = scores

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        passages = request.passages

        for passage, score in zip(passages, self.scores):
            passage["score"] = score

        passages.sort(key=lambda item: item["score"], reverse=True)
        return passages


def make_candidates() -> list[dict[str, Any]]:
    """Build candidates with source metadata."""
    return [
        {
            "id": "doc-1:0",
            "text": "Unrelated passage",
            "document_id": "doc-1",
            "source_name": "sample.pdf",
            "chunk_index": 0,
            "page_numbers": (2,),
            "distance": 0.8,
        },
        {
            "id": "doc-1:1",
            "text": "Relevant passage",
            "document_id": "doc-1",
            "source_name": "sample.pdf",
            "chunk_index": 1,
            "page_numbers": (3, 4),
            "distance": 0.2,
        },
    ]


def test_rerank_changes_order_using_model_scores() -> None:
    candidates = make_candidates()
    original_candidates = [dict(item) for item in candidates]
    reranker = PassageReranker(
        ranker=FakeRanker([0.01, 0.99])
    )

    results = reranker.rerank("question", candidates)

    assert [item["id"] for item in results] == ["doc-1:1", "doc-1:0"]
    assert results[0]["score"] == pytest.approx(0.99)
    assert results[1]["score"] == pytest.approx(0.01)
    assert candidates == original_candidates


def test_rerank_preserves_passage_metadata() -> None:
    candidates = make_candidates()
    reranker = PassageReranker(
        ranker=FakeRanker([0.1, 0.9])
    )

    results = reranker.rerank("question", candidates)
    relevant = results[0]

    assert relevant["text"] == "Relevant passage"
    assert relevant["document_id"] == "doc-1"
    assert relevant["source_name"] == "sample.pdf"
    assert relevant["chunk_index"] == 1
    assert relevant["page_numbers"] == (3, 4)
    assert relevant["distance"] == 0.2
    assert relevant["score"] == pytest.approx(0.9)


def test_rerank_returns_empty_list_for_empty_candidates() -> None:
    reranker = PassageReranker(ranker=FakeRanker([]))

    assert reranker.rerank("question", []) == []


@pytest.mark.parametrize("query", ["", "   ", None])
def test_rerank_rejects_invalid_query(query: Any) -> None:
    reranker = PassageReranker(ranker=FakeRanker([]))

    with pytest.raises(ValueError, match="non-empty string"):
        reranker.rerank(query, [])


@pytest.mark.parametrize(
    "candidates",
    [
        "invalid",
        [None],
        [{"id": "missing-text"}],
        [{"text": ""}],
        [{"text": "   "}],
    ],
)
def test_rerank_rejects_invalid_candidates(candidates: Any) -> None:
    reranker = PassageReranker(ranker=FakeRanker([]))

    with pytest.raises(ValueError):
        reranker.rerank("question", candidates)


@pytest.mark.parametrize("max_length", [0, -1, 1.5, True])
def test_reranker_rejects_invalid_max_length(max_length: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PassageReranker(max_length=max_length)


def test_rerank_rejects_invalid_model_scores() -> None:
    class InvalidScoreRanker:
        def rerank(self, request: Any) -> list[dict[str, Any]]:
            return [{**request.passages[0], "score": float("nan")}]

    reranker = PassageReranker(ranker=InvalidScoreRanker())

    with pytest.raises(RuntimeError, match="invalid relevance score"):
        reranker.rerank("question", [{"text": "A passage"}])


def test_count_tokens_uses_ranker_tokenizer() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> Any:
            return type("Encoding", (), {"ids": [1, 2, 3]})()

    fake_ranker = type(
        "FakeRankerWithTokenizer",
        (),
        {"tokenizer": FakeTokenizer()},
    )()

    reranker = PassageReranker(ranker=fake_ranker)

    assert reranker.count_tokens("sample text") == 3


def test_count_tokens_rejects_non_string_input() -> None:
    reranker = PassageReranker(ranker=FakeRanker([]))

    with pytest.raises(ValueError, match="must be a string"):
        reranker.count_tokens(123)


def test_reranker_wraps_model_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_ranker(**kwargs: Any) -> Any:
        raise OSError("Model initialization failed.")

    flashrank_module = ModuleType("flashrank")
    flashrank_module.Ranker = failing_ranker
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)

    reranker = PassageReranker()

    with pytest.raises(
        RuntimeError,
        match="Failed to initialize the FlashRank reranking model",
    ) as exc_info:
        reranker._get_ranker()

    assert isinstance(exc_info.value.__cause__, OSError)


def test_reranker_wraps_ranking_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRerankRequest:
        def __init__(self, query: str, passages: list[dict[str, Any]]) -> None:
            self.query = query
            self.passages = passages

    class FailingRanker:
        def rerank(self, request: Any) -> Any:
            raise OSError("Ranking execution failed.")

    flashrank_module = ModuleType("flashrank")
    flashrank_module.RerankRequest = FakeRerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", flashrank_module)

    reranker = PassageReranker(ranker=FailingRanker())

    with pytest.raises(
        RuntimeError,
        match="Failed to rerank document passages",
    ) as exc_info:
        reranker.rerank("question", make_candidates())

    assert isinstance(exc_info.value.__cause__, OSError)
