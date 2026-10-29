"""Semantic retrieval from the active document index."""

from typing import Any, Mapping

from pdf_rag.embeddings import EmbeddingModel
from pdf_rag.indexing import SESSION_INDEX_KEY, DocumentIndex


DEFAULT_RETRIEVAL_CANDIDATES = 20


def retrieve_chunks(
    query: str,
    session_state: Mapping[str, object],
    embedding_model: EmbeddingModel | None = None,
    n_results: int = DEFAULT_RETRIEVAL_CANDIDATES,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks from the active document index.

    Args:
        query: User's natural-language question.
        session_state: Session state containing the active DocumentIndex.
        embedding_model: Optional embedding model, useful for testing.
        n_results: Maximum number of candidate chunks to return.

    Returns:
        Ranked chunks with their text, source metadata, page numbers,
        and cosine distance.

    Raises:
        ValueError: If the query is empty or n_results is invalid.
        RuntimeError: If no valid document index is active or retrieval fails.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if isinstance(n_results, bool) or not isinstance(n_results, int):
        raise ValueError("Number of results must be a positive integer.")

    if n_results < 1:
        raise ValueError("Number of results must be a positive integer.")

    active_index = session_state.get(SESSION_INDEX_KEY)

    if not isinstance(active_index, DocumentIndex):
        raise RuntimeError(
            "No active document index. Index a PDF before retrieving chunks."
        )

    if active_index.store.count() == 0:
        return []

    model = embedding_model or EmbeddingModel()

    try:
        query_embeddings = model.embed_texts([query])
        if len(query_embeddings) != 1:
            raise RuntimeError(
                "The embedding model must return exactly one query embedding."
            )

        results = active_index.store.query(
            query_embeddings[0],
            n_results=n_results,
        )
    except Exception as exc:
        raise RuntimeError("Failed to retrieve document chunks.") from exc

    retrieved_chunks: list[dict[str, Any]] = []

    for result in results:
        metadata = result["metadata"]
        raw_pages = metadata.get("page_numbers", "")

        page_numbers = tuple(
            int(page.strip())
            for page in raw_pages.split(",")
            if page.strip()
        )

        retrieved_chunks.append(
            {
                "id": result["id"],
                "text": result["text"],
                "document_id": metadata["document_id"],
                "source_name": metadata["source_name"],
                "chunk_index": metadata["chunk_index"],
                "page_numbers": page_numbers,
                "distance": result["distance"],
            }
        )

    return retrieved_chunks
