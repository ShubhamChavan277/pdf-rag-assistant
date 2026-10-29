"""Tests for embedding, vector storage, and active document indexing."""

from typing import Sequence

import pytest

from pdf_rag.embeddings import EmbeddingModel
from pdf_rag.indexing import (
    SESSION_INDEX_KEY,
    DocumentIndex,
    index_document,
)
from pdf_rag.ingestion_models import ChunkRecord
from pdf_rag.vector_store import VectorStore


class FakeEmbeddingModel:
    """Deterministic embedding model for isolated tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        self.calls.append(values)
        return [
            [float(len(text)), float(sum(ord(char) for char in text) % 101 + 1)]
            for text in values
        ]


class FakeSentenceTransformer:
    """Minimal stand-in for SentenceTransformer."""

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        import numpy as np

        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False

        return np.array([[0.6, 0.8] for _ in texts], dtype=np.float32)


def make_chunks(
    document_id: str = "doc-1",
    source_name: str = "sample.pdf",
    texts: tuple[str, ...] = ("First paragraph.", "Second paragraph."),
) -> list[ChunkRecord]:
    """Build representative chunk records."""
    return [
        ChunkRecord(
            document_id=document_id,
            source_name=source_name,
            chunk_index=index,
            text=text,
            page_numbers=(index + 1,),
        )
        for index, text in enumerate(texts)
    ]


def test_embedding_model_normalizes_and_returns_float_vectors() -> None:
    model = EmbeddingModel(model=FakeSentenceTransformer())

    vectors = model.embed_texts(["sample text"])

    assert len(vectors) == 1
    assert vectors[0] == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("texts", [[], [""], ["   "], ["valid", ""]])
def test_embedding_model_rejects_empty_inputs(texts: list[str]) -> None:
    model = EmbeddingModel(model=FakeSentenceTransformer())

    with pytest.raises(ValueError):
        model.embed_texts(texts)


def test_vector_store_saves_text_and_metadata() -> None:
    chunks = make_chunks()
    store = VectorStore()

    store.add_chunks(
        chunks,
        [[0.1, 0.2], [0.3, 0.4]],
    )

    assert store.count() == 2

    result = store._collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    assert set(result["documents"]) == {
        "First paragraph.",
        "Second paragraph.",
    }

    records = {
        metadata["chunk_index"]: metadata
        for metadata in result["metadatas"]
    }
    assert records[0]["document_id"] == "doc-1"
    assert records[0]["source_name"] == "sample.pdf"
    assert records[0]["page_numbers"] == "1"
    assert records[1]["page_numbers"] == "2"


def test_vector_store_rejects_mismatched_chunk_and_embedding_counts() -> None:
    store = VectorStore()

    with pytest.raises(ValueError, match="exactly one embedding"):
        store.add_chunks(make_chunks(), [[0.1, 0.2]])


def test_indexing_reuses_unchanged_document_without_reembedding() -> None:
    state: dict[str, object] = {}
    model = FakeEmbeddingModel()
    chunks = make_chunks()

    first = index_document(chunks, state, model)
    second = index_document(chunks, state, model)

    assert isinstance(first, DocumentIndex)
    assert second is first
    assert model.calls == [
        ["First paragraph.", "Second paragraph."]
    ]
    assert first.store.count() == 2


def test_indexing_replaces_old_document_content() -> None:
    state: dict[str, object] = {}
    model = FakeEmbeddingModel()

    first = index_document(make_chunks(), state, model)
    old_store = first.store

    replacement_chunks = make_chunks(
        document_id="doc-2",
        source_name="replacement.pdf",
        texts=("Replacement content.",),
    )
    second = index_document(replacement_chunks, state, model)

    assert second is not first
    assert second.chunk_count == 1
    assert second.store.count() == 1
    assert old_store.count() == 0

    stored = second.store._collection.get(include=["documents", "metadatas"])
    assert stored["documents"] == ["Replacement content."]
    assert stored["metadatas"][0]["document_id"] == "doc-2"
    assert stored["metadatas"][0]["source_name"] == "replacement.pdf"


def test_indexing_rebuilds_when_session_state_is_missing_or_reset() -> None:
    model = FakeEmbeddingModel()
    chunks = make_chunks()
    state: dict[str, object] = {}

    first = index_document(chunks, state, model)
    del state[SESSION_INDEX_KEY]

    second = index_document(chunks, state, model)

    assert second is not first
    assert second.store.count() == 2
    assert len(model.calls) == 2


def test_indexing_rebuilds_when_active_store_is_incomplete() -> None:
    state: dict[str, object] = {}
    model = FakeEmbeddingModel()
    chunks = make_chunks()

    first = index_document(chunks, state, model)
    first.store.clear()

    second = index_document(chunks, state, model)

    assert second is not first
    assert second.store.count() == len(chunks)
    assert len(model.calls) == 2
