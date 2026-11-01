
import pytest

from pdf_rag.evaluation_metrics import aggregate_metrics, evaluate_retrieval


@pytest.fixture
def sample_results():
    return [
        {
            "source_name": "irctc.pdf",
            "page_numbers": [1],
        },
        {
            "source_name": "traffic.pdf",
            "page_numbers": [2],
        },
        {
            "source_name": "irctc.pdf",
            "page_numbers": [2, 3],
        },
        {
            "source_name": "irctc.pdf",
            "page_numbers": [4],
        },
        {
            "source_name": "traffic.pdf",
            "page_numbers": [1],
        },
    ]


def test_evaluate_retrieval_calculates_metrics(sample_results):
    scores = evaluate_retrieval(
        source_document="irctc.pdf",
        relevant_pages=[2, 3],
        ranked_results=sample_results,
        k=5,
    )

    assert scores == {
        "hit_at_k": 1.0,
        "recall_at_k": 1.0,
        "precision_at_k": 0.2,
        "reciprocal_rank": 1 / 3,
    }


def test_evaluate_retrieval_returns_zero_when_no_match(sample_results):
    scores = evaluate_retrieval(
        source_document="missing.pdf",
        relevant_pages=[1],
        ranked_results=sample_results,
        k=5,
    )

    assert scores == {
        "hit_at_k": 0.0,
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_evaluate_retrieval_handles_fewer_than_k_results():
    results = [
        {"source_name": "irctc.pdf", "page_numbers": [1]}
    ]

    scores = evaluate_retrieval(
        source_document="irctc.pdf",
        relevant_pages=[1],
        ranked_results=results,
        k=5,
    )

    assert scores["hit_at_k"] == 1.0
    assert scores["recall_at_k"] == 1.0
    assert scores["precision_at_k"] == 0.2
    assert scores["reciprocal_rank"] == 1.0


def test_evaluate_retrieval_rejects_invalid_k(sample_results):
    with pytest.raises(ValueError):
        evaluate_retrieval(
            source_document="irctc.pdf",
            relevant_pages=[1],
            ranked_results=sample_results,
            k=0,
        )


def test_evaluate_retrieval_rejects_empty_gold_pages(sample_results):
    with pytest.raises(ValueError):
        evaluate_retrieval(
            source_document="irctc.pdf",
            relevant_pages=[],
            ranked_results=sample_results,
        )


def test_aggregate_metrics_calculates_macro_averages():
    scores = [
        {
            "hit_at_k": 1.0,
            "recall_at_k": 1.0,
            "precision_at_k": 0.4,
            "reciprocal_rank": 1.0,
        },
        {
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "reciprocal_rank": 0.0,
        },
    ]

    result = aggregate_metrics(scores)

    assert result == {
        "hit_at_k": 0.5,
        "recall_at_k": 0.5,
        "precision_at_k": 0.2,
        "reciprocal_rank": 0.5,
    }


def test_aggregate_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_metrics([])
