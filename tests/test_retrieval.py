"""Tests for semantic retrieval from the active document index."""

from collections.abc import Sequence

import pytest

from pdf_rag.indexing import SESSION_INDEX_KEY, DocumentIndex, index_document
from pdf_rag.ingestion_models import ChunkRecord
from pdf_rag.retrieval import retrieve_chunks
from pdf_rag.vector_store import VectorStore


class FakeEmbeddingModel:
    """Return deterministic vectors without loading a real model."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = {
            "question": [1.0, 0.0],
            "Matching passage": [1.0, 0.0],
            "Unrelated passage": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def make_chunks() -> list[ChunkRecord]:
    """Build chunks with known content and source pages."""
    return [
        ChunkRecord(
            document_id="doc-1",
            source_name="sample.pdf",
            chunk_index=0,
            text="Matching passage",
            page_numbers=(2, 3),
        ),
        ChunkRecord(
            document_id="doc-1",
            source_name="sample.pdf",
            chunk_index=1,
            text="Unrelated passage",
            page_numbers=(4,),
        ),
    ]


def make_index() -> tuple[dict[str, object], FakeEmbeddingModel]:
    """Build an active index using deterministic embeddings."""
    state: dict[str, object] = {}
    model = FakeEmbeddingModel()
    index_document(make_chunks(), state, model)
    return state, model


def test_retrieve_chunks_returns_ranked_results_with_source_metadata() -> None:
    state, model = make_index()

    results = retrieve_chunks("question", state, model)

    assert len(results) == 2
    assert results[0]["text"] == "Matching passage"
    assert results[0]["id"] == "doc-1:0"
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["source_name"] == "sample.pdf"
    assert results[0]["chunk_index"] == 0
    assert results[0]["page_numbers"] == (2, 3)
    assert results[0]["distance"] == pytest.approx(0.0)
    assert results[0]["distance"] <= results[1]["distance"]


def test_retrieve_chunks_limits_number_of_results() -> None:
    state, model = make_index()

    results = retrieve_chunks("question", state, model, n_results=1)

    assert len(results) == 1
    assert results[0]["text"] == "Matching passage"


def test_retrieve_chunks_rejects_empty_query() -> None:
    state, model = make_index()

    for query in ("", "   "):
        with pytest.raises(ValueError, match="non-empty string"):
            retrieve_chunks(query, state, model)


@pytest.mark.parametrize("n_results", [0, -1, 1.5, True])
def test_retrieve_chunks_rejects_invalid_candidate_count(
    n_results: int | float | bool,
) -> None:
    state, model = make_index()

    with pytest.raises(ValueError, match="positive integer"):
        retrieve_chunks("question", state, model, n_results=n_results)


def test_retrieve_chunks_requires_active_document_index() -> None:
    with pytest.raises(RuntimeError, match="No active document index"):
        retrieve_chunks("question", {}, FakeEmbeddingModel())


def test_retrieve_chunks_returns_empty_list_for_empty_store() -> None:
    state: dict[str, object] = {
        SESSION_INDEX_KEY: DocumentIndex(
            fingerprint="empty",
            store=VectorStore(),
            chunk_count=0,
        )
    }

    results = retrieve_chunks("question", state, FakeEmbeddingModel())

    assert results == []
