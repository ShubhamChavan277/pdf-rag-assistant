"""In-memory ChromaDB storage for indexed PDF chunks."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import chromadb

from pdf_rag.ingestion_models import ChunkRecord


DEFAULT_COLLECTION_NAME = "pdf_document_chunks"


class VectorStore:
    """Store chunk text, embeddings, IDs, and provenance in ChromaDB."""

    def __init__(
        self,
        client: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Create an isolated in-memory collection unless one is named."""
        if collection_name is None:
            collection_name = f"{DEFAULT_COLLECTION_NAME}_{uuid4().hex}"

        if not collection_name.strip():
            raise ValueError("Collection name must not be empty.")

        self._client = client if client is not None else chromadb.EphemeralClient()
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_id(chunk: ChunkRecord) -> str:
        """Return a stable ID for a chunk within its document."""
        return f"{chunk.document_id}:{chunk.chunk_index}"

    def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store chunks with precomputed embeddings and source metadata."""
        if not chunks:
            raise ValueError("At least one chunk must be provided.")

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        ids = [self._chunk_id(chunk) for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Chunk IDs must be unique.")

        vectors: list[list[float]] = []
        for embedding in embeddings:
            vector = [float(value) for value in embedding]

            if not vector:
                raise ValueError("Embeddings must not be empty.")

            vectors.append(vector)

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimension.")

        self._collection.add(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "source_name": chunk.source_name,
                    "chunk_index": chunk.chunk_index,
                    "page_numbers": ",".join(
                        str(number) for number in chunk.page_numbers
                    ),
                }
                for chunk in chunks
            ],
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete and recreate the collection to remove stale chunks."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
