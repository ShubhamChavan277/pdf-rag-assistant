"""Sentence Transformer embedding generation."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate normalized embeddings using a reusable Sentence Transformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model: Any | None = None,
    ) -> None:
        """Initialize lazily, allowing a model to be injected for tests."""
        self.model_name = model_name
        self._model = model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert non-empty text inputs into normalized embedding vectors."""
        if not texts:
            raise ValueError("At least one text must be provided.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("The embedding model returned an invalid shape.")

        if not np.isfinite(vectors).all():
            raise ValueError("The embedding model returned non-finite values.")

        return vectors.tolist()
