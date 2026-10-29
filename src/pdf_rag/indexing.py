"""Document indexing and active-index lifecycle management."""

import hashlib
import json
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass

from pdf_rag.embeddings import EmbeddingModel
from pdf_rag.ingestion_models import ChunkRecord
from pdf_rag.vector_store import VectorStore


SESSION_INDEX_KEY = "pdf_rag_document_index"


@dataclass
class DocumentIndex:
    """An indexed document and its source-content fingerprint."""

    fingerprint: str
    store: VectorStore
    chunk_count: int


def _fingerprint_chunks(chunks: Sequence[ChunkRecord]) -> str:
    """Calculate a deterministic fingerprint of chunk content and metadata."""
    payload = [
        {
            "document_id": chunk.document_id,
            "source_name": chunk.source_name,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "page_numbers": list(chunk.page_numbers),
        }
        for chunk in chunks
    ]

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def index_document(
    chunks: Sequence[ChunkRecord],
    session_state: MutableMapping[str, object],
    embedding_model: EmbeddingModel | None = None,
) -> DocumentIndex:
    """Index chunks, reusing an unchanged and intact active index."""
    if not chunks:
        raise ValueError("At least one chunk is required for indexing.")

    fingerprint = _fingerprint_chunks(chunks)
    existing = session_state.get(SESSION_INDEX_KEY)

    if (
        isinstance(existing, DocumentIndex)
        and existing.fingerprint == fingerprint
        and existing.chunk_count == len(chunks)
        and existing.store.count() == len(chunks)
    ):
        return existing

    # Remove the active reference before rebuilding. If embedding or storage
    # fails, the previous document must not remain active.
    session_state.pop(SESSION_INDEX_KEY, None)

    if isinstance(existing, DocumentIndex):
        existing.store.clear()

    model = embedding_model if embedding_model is not None else EmbeddingModel()
    embeddings = model.embed_texts([chunk.text for chunk in chunks])

    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    if store.count() != len(chunks):
        store.clear()
        raise RuntimeError("Indexed chunk count does not match input.")

    index = DocumentIndex(
        fingerprint=fingerprint,
        store=store,
        chunk_count=len(chunks),
    )
    session_state[SESSION_INDEX_KEY] = index

    return index
