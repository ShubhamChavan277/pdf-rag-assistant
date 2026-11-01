
"""Pure-Python retrieval evaluation metrics for the M8 benchmark."""

from collections.abc import Mapping, Sequence
from typing import Any


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _is_relevant(
    result: Mapping[str, Any],
    source_document: str,
    relevant_pages: set[int],
) -> bool:
    """Check relevance using source filename and page provenance."""
    if result.get("source_name") != source_document:
        return False

    result_pages = result.get("page_numbers", [])

    if not isinstance(result_pages, (list, tuple, set)):
        return False

    return bool(relevant_pages.intersection(result_pages))


def evaluate_retrieval(
    source_document: str,
    relevant_pages: Sequence[int],
    ranked_results: Sequence[Mapping[str, Any]],
    k: int = 5,
) -> dict[str, float]:
    """Calculate retrieval metrics for one answerable question.

    Recall@k measures the proportion of gold pages represented in the
    top-k results. Precision@k measures the proportion of top-k chunks
    that match the gold document and at least one gold page.
    """
    _validate_k(k)

    if not source_document:
        raise ValueError("source_document must not be empty.")

    gold_pages = set(relevant_pages)

    if not gold_pages or any(
        not isinstance(page, int) or isinstance(page, bool) or page <= 0
        for page in gold_pages
    ):
        raise ValueError("relevant_pages must contain positive page numbers.")

    top_results = list(ranked_results[:k])
    relevance = [
        _is_relevant(result, source_document, gold_pages)
        for result in top_results
    ]

    relevant_pages_found = {
        page
        for result in top_results
        if result.get("source_name") == source_document
        for page in result.get("page_numbers", [])
        if page in gold_pages
    }

    first_relevant_rank = next(
        (rank for rank, is_relevant in enumerate(relevance, start=1)
         if is_relevant),
        None,
    )

    return {
        "hit_at_k": float(any(relevance)),
        "recall_at_k": len(relevant_pages_found) / len(gold_pages),
        "precision_at_k": sum(relevance) / k,
        "reciprocal_rank": (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
    }


def aggregate_metrics(
    question_scores: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Calculate macro averages across evaluated questions."""
    if not question_scores:
        raise ValueError("At least one question score is required.")

    metric_names = (
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "reciprocal_rank",
    )

    return {
        name: sum(score[name] for score in question_scores)
        / len(question_scores)
        for name in metric_names
    }
