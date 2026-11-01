"""Run the PDF parser and retrieval evaluation benchmark.

Run from the project root:
    python evaluation/run_evaluation.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pdf_rag.chunking import chunk_pages
from pdf_rag.embeddings import EmbeddingModel
from pdf_rag.evaluation_metrics import aggregate_metrics, evaluate_retrieval
from pdf_rag.indexing import index_document
from pdf_rag.pdfplumber_adapter import extract_pages_pdfplumber
from pdf_rag.pypdf_adapter import extract_pages_pypdf
from pdf_rag.reranking import PassageReranker
from pdf_rag.retrieval import retrieve_chunks
from pdf_rag.answer_generation import (
    ABSTENTION_MESSAGE,
    AnswerGenerator,
)
from pdf_rag.llm_config import load_groq_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
DOCUMENTS_DIR = PROJECT_ROOT / "evaluation" / "documents"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

TOP_K = 5
RETRIEVAL_CANDIDATES = 20


def load_dataset() -> dict[str, Any]:
    """Load and perform basic structural validation of the benchmark."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset.get("documents"), list):
        raise ValueError("Dataset must contain a documents list.")

    if not isinstance(dataset.get("questions"), list):
        raise ValueError("Dataset must contain a questions list.")

    question_ids = [
        question.get("id")
        for question in dataset["questions"]
    ]

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Dataset question IDs must be unique.")

    return dataset


def normalize_text(text: str) -> str:
    """Normalize whitespace for parser text comparison."""
    return re.sub(r"\s+", " ", text).strip()


def compare_parsers(
    document_paths: list[Path],
) -> dict[str, Any]:
    """Compare parser output statistics and normalized page similarity."""
    comparisons: dict[str, Any] = {}

    for document_path in document_paths:
        document_name = document_path.name
        document_result: dict[str, Any] = {}

        for parser_name, extract_function in (
            ("pypdf", extract_pages_pypdf),
            ("pdfplumber", extract_pages_pdfplumber),
        ):
            started = time.perf_counter()
            pages = extract_function(document_path)
            elapsed = time.perf_counter() - started

            document_result[parser_name] = {
                "page_count": len(pages),
                "character_count": sum(len(page.text) for page in pages),
                "empty_pages": sum(
                    not page.text.strip() for page in pages
                ),
                "elapsed_seconds": round(elapsed, 6),
                "_pages": pages,
            }

        pypdf_pages = document_result["pypdf"].pop("_pages")
        pdfplumber_pages = document_result["pdfplumber"].pop("_pages")

        if len(pypdf_pages) != len(pdfplumber_pages):
            similarities = []
        else:
            similarities = [
                SequenceMatcher(
                    None,
                    normalize_text(pypdf_page.text),
                    normalize_text(pdfplumber_page.text),
                ).ratio()
                for pypdf_page, pdfplumber_page
                in zip(pypdf_pages, pdfplumber_pages)
            ]

        document_result["normalized_page_similarity"] = [
            round(score, 4) for score in similarities
        ]
        document_result["mean_page_similarity"] = (
            round(statistics.mean(similarities), 4)
            if similarities
            else None
        )

        comparisons[document_name] = document_result

    return comparisons


def summarize_latencies(values: list[float]) -> dict[str, float]:
    """Summarize durations in milliseconds."""
    if not values:
        return {}

    ordered = sorted(values)
    percentile_index = max(0, math.ceil(0.95 * len(ordered)) - 1)

    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values) * 1000, 3),
        "median_ms": round(statistics.median(values) * 1000, 3),
        "p95_ms": round(ordered[percentile_index] * 1000, 3),
        "min_ms": round(min(values) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
    }


def serialize_rankings(
    ranked_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep useful, JSON-safe retrieval evidence in the raw results."""
    serialized = []

    for rank, result in enumerate(ranked_results, start=1):
        item = {
            "rank": rank,
            "source_name": result.get("source_name"),
            "page_numbers": list(result.get("page_numbers", [])),
            "chunk_index": result.get("chunk_index"),
            "text": result.get("text", ""),
        }

        if "distance" in result:
            item["distance"] = result["distance"]

        if "score" in result:
            item["reranker_score"] = result["score"]

        serialized.append(item)

    return serialized


def run_generation_evaluation(
    dataset: dict[str, Any],
    session_state: dict[str, Any],
    embedding_model: EmbeddingModel,
    reranker: PassageReranker,
) -> dict[str, Any]:
    """Evaluate generation on two answerable and four unanswerable questions."""
    answerable_ids = {"I01", "T01"}

    selected_questions = [
        question
        for question in dataset["questions"]
        if (
            question["id"] in answerable_ids
            or question.get("expected_answer") is None
        )
    ]

    if len(selected_questions) != 6:
        raise ValueError(
            "Generation evaluation requires I01, T01, and U01-U04."
        )

    # Enforce zero automatic retries for this evaluation only.
    config = replace(load_groq_config(), max_retries=0)
    generator = AnswerGenerator(config)

    generation_results = []

    for question_index, question in enumerate(selected_questions):
        question_id = question["id"]
        query = question["question"]

        try:
            candidates = retrieve_chunks(
                query,
                session_state,
                embedding_model=embedding_model,
                n_results=RETRIEVAL_CANDIDATES,
            )

            ranked_passages = reranker.rerank(query, candidates)

            generation_started = time.perf_counter()
            answer_result = generator.generate_answer(
                query,
                ranked_passages,
            )
            generation_seconds = time.perf_counter() - generation_started

            generation_results.append({
                "question_id": question_id,
                "question": query,
                "answerable": question.get("expected_answer") is not None,
                "answer": answer_result.answer,
                "exact_abstention_message": (
                    answer_result.answer == ABSTENTION_MESSAGE
                ),
                "generation_seconds": round(generation_seconds, 6),
                "sources": serialize_rankings(
                    list(answer_result.sources)
                ),
                "error": None,
            })

            print(
                f"{question_id}: generation completed "
                f"({generation_seconds:.2f}s)"
            )

        except Exception as exc:
            generation_results.append({
                "question_id": question_id,
                "question": query,
                "answerable": question.get("expected_answer") is not None,
                "answer": None,
                "exact_abstention_message": None,
                "generation_seconds": None,
                "sources": [],
                "error": type(exc).__name__,
            })

            print(
                f"{question_id}: generation failed "
                f"({type(exc).__name__})"
            )

        if question_index < len(selected_questions) - 1:
            time.sleep(35)

    successful_results = [
        result
        for result in generation_results
        if result["generation_seconds"] is not None
    ]

    return {
        "enabled": True,
        "model": config.model,
        "configured_max_retries": config.max_retries,
        "planned_api_calls": len(selected_questions),
        "successful_generations": len(successful_results),
        "failed_generations": (
            len(generation_results) - len(successful_results)
        ),
        "latency": summarize_latencies([
            result["generation_seconds"]
            for result in successful_results
        ]),
        "results": generation_results,
        "limitations": [
            "Only two answerable and four unanswerable questions "
            "were selected for generation evaluation.",
            "No automated LLM judge was used.",
            "Exact abstention-message matching does not capture "
            "semantically equivalent abstentions.",
            "Generated answers and sources require manual review.",
        ],
    }


def run_evaluation(generate: bool = False) -> Path:
    """Run parser comparison and baseline/reranked retrieval evaluation."""
    dataset = load_dataset()

    document_paths = [
        DOCUMENTS_DIR / document_name
        for document_name in dataset["documents"]
    ]

    missing_documents = [
        path.name for path in document_paths if not path.is_file()
    ]

    if missing_documents:
        raise FileNotFoundError(
            "Missing evaluation PDF(s): " + ", ".join(missing_documents)
        )

    answerable_questions = [
        question
        for question in dataset["questions"]
        if question.get("expected_answer") is not None
    ]

    unanswerable_questions = [
        question
        for question in dataset["questions"]
        if question.get("expected_answer") is None
    ]

    if not answerable_questions:
        raise ValueError("Dataset contains no answerable questions.")

    print("PDF RAG evaluation started")
    print(f"Documents: {len(document_paths)}")
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: "
          f"{len(dataset['questions']) - len(answerable_questions)}")

    # Parser comparison is independent of retrieval indexing.
    parser_results = compare_parsers(document_paths)

    # Use pypdf as the consistent parser for the retrieval benchmark.
    all_chunks = []
    chunk_statistics = {}

    for document_path in document_paths:
        pages = extract_pages_pypdf(document_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

        chunk_statistics[document_path.name] = {
            "pages": len(pages),
            "chunks": len(chunks),
            "average_chunk_characters": round(
                statistics.mean(len(chunk.text) for chunk in chunks), 2
            ) if chunks else 0,
            "minimum_chunk_characters": min(
                (len(chunk.text) for chunk in chunks), default=0
            ),
            "maximum_chunk_characters": max(
                (len(chunk.text) for chunk in chunks), default=0
            ),
        }

    # Load one embedding model and reuse it for indexing and retrieval.
    embedding_model = EmbeddingModel()
    session_state: dict[str, Any] = {}

    indexing_started = time.perf_counter()
    index_document(
        all_chunks,
        session_state,
        embedding_model=embedding_model,
    )
    indexing_seconds = time.perf_counter() - indexing_started

    # Reuse one reranker instance for every question.
    reranker = PassageReranker()

    baseline_scores = []
    reranked_scores = []
    question_results = []

    retrieval_latencies = []
    reranking_latencies = []

    reranker_first_call_seconds = None

    for question in answerable_questions:
        question_id = question["id"]
        query = question["question"]
        source_document = question["source_document"]
        relevant_pages = question["relevant_pages"]

        retrieval_started = time.perf_counter()
        candidates = retrieve_chunks(
            query,
            session_state,
            embedding_model=embedding_model,
            n_results=RETRIEVAL_CANDIDATES,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        retrieval_latencies.append(retrieval_seconds)

        reranking_started = time.perf_counter()
        reranked = reranker.rerank(query, candidates)
        reranking_seconds = time.perf_counter() - reranking_started
        reranking_latencies.append(reranking_seconds)

        if reranker_first_call_seconds is None:
            reranker_first_call_seconds = reranking_seconds

        baseline_metrics = evaluate_retrieval(
            source_document,
            relevant_pages,
            candidates,
            k=TOP_K,
        )
        reranked_metrics = evaluate_retrieval(
            source_document,
            relevant_pages,
            reranked,
            k=TOP_K,
        )

        baseline_scores.append(baseline_metrics)
        reranked_scores.append(reranked_metrics)

        question_results.append({
            "question_id": question_id,
            "question": query,
            "expected_source_document": source_document,
            "relevant_pages": relevant_pages,
            "candidate_count": len(candidates),
            "baseline_metrics": baseline_metrics,
            "reranked_metrics": reranked_metrics,
            "retrieval_seconds": round(retrieval_seconds, 6),
            "reranking_seconds": round(reranking_seconds, 6),
            "baseline_rankings": serialize_rankings(candidates),
            "reranked_rankings": serialize_rankings(reranked),
        })

        print(
            f"{question_id}: "
            f"baseline MRR={baseline_metrics['reciprocal_rank']:.3f}, "
            f"reranked MRR={reranked_metrics['reciprocal_rank']:.3f}"
        )

    # Record retrieval diagnostics for unanswerable questions.
    # No relevance metrics are calculated because these questions
    # deliberately have no gold document or relevant pages.
    unanswerable_results = []

    for question in unanswerable_questions:
        question_id = question["id"]
        query = question["question"]

        retrieval_started = time.perf_counter()
        candidates = retrieve_chunks(
            query,
            session_state,
            embedding_model=embedding_model,
            n_results=RETRIEVAL_CANDIDATES,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started

        reranking_started = time.perf_counter()
        reranked = reranker.rerank(query, candidates)
        reranking_seconds = time.perf_counter() - reranking_started

        unanswerable_results.append({
            "question_id": question_id,
            "question": query,
            "candidate_count": len(candidates),
            "retrieval_seconds": round(retrieval_seconds, 6),
            "reranking_seconds": round(reranking_seconds, 6),
            "baseline_rankings": serialize_rankings(candidates),
            "reranked_rankings": serialize_rankings(reranked),
        })

        print(
            f"{question_id}: unanswerable diagnostic recorded "
            f"({len(candidates)} candidates)"
        )

    generation_results = (
        run_generation_evaluation(
            dataset,
            session_state,
            embedding_model,
            reranker,
        )
        if generate
        else {
            "enabled": False,
            "reason": "Generation evaluation requires --generate.",
        }
    )

    results = {
        "benchmark": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "parser_for_retrieval": "pypdf",
            "top_k": TOP_K,
            "retrieval_candidates": RETRIEVAL_CANDIDATES,
            "answerable_question_count": len(answerable_questions),
            "generation_evaluation_enabled": generate,
        },
        "parser_comparison": parser_results,
        "chunk_statistics": chunk_statistics,
        "indexing": {
            "total_chunks": len(all_chunks),
            "elapsed_seconds": round(indexing_seconds, 6),
        },
        "latency": {
            "retrieval": summarize_latencies(retrieval_latencies),
            "reranking": summarize_latencies(reranking_latencies),
            "first_reranking_call_ms": (
                round(reranker_first_call_seconds * 1000, 3)
                if reranker_first_call_seconds is not None
                else None
            ),
        },
        "aggregate_metrics": {
            "baseline": aggregate_metrics(baseline_scores),
            "reranked": aggregate_metrics(reranked_scores),
        },
        "per_question_results": question_results,
        "unanswerable_diagnostics": unanswerable_results,
        "generation_evaluation": generation_results,
        "limitations": [
            "Metrics are calculated only for answerable questions.",
            "Recall measures gold-page coverage; precision measures "
            "relevant chunks among the top-k positions.",
            "Parser timings are single-run measurements, not statistical "
            "performance benchmarks.",
            "Indexing time includes embedding and vector-store construction.",
            "Generation evaluation was not run.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"retrieval_results_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\nAggregate retrieval metrics")
    print("Baseline:", results["aggregate_metrics"]["baseline"])
    print("Reranked:", results["aggregate_metrics"]["reranked"])
    print(f"\nRaw results saved to: {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the PDF RAG evaluation benchmark."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Enable the optional six-question Groq generation evaluation.",
    )
    arguments = parser.parse_args()

    run_evaluation(generate=arguments.generate)
