"""Cross-encoder reranking of retrieved document passages."""

import math
import numbers
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_RERANKING_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RERANKING_MAX_LENGTH = 512


def get_default_reranking_cache_dir() -> Path:
    """Return a model cache directory outside the project repository."""
    local_app_data = Path(
        os.environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
    )
    return local_app_data / "PDF-RAG-Assistant" / "flashrank"


class PassageReranker:
    """Rerank retrieved passages using a FlashRank cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKING_MODEL,
        cache_dir: str | Path | None = None,
        max_length: int = DEFAULT_RERANKING_MAX_LENGTH,
        ranker: Any | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise ValueError("Maximum input length must be a positive integer.")

        if max_length < 1:
            raise ValueError("Maximum input length must be a positive integer.")

        self.model_name = model_name
        self.cache_dir = Path(
            cache_dir if cache_dir is not None
            else get_default_reranking_cache_dir()
        )
        self.max_length = max_length
        self._ranker = ranker

    def _get_ranker(self) -> Any:
        """Initialize FlashRank lazily, downloading the model if necessary."""
        if self._ranker is None:
            try:
                from flashrank import Ranker

                self._ranker = Ranker(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir),
                    max_length=self.max_length,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the FlashRank reranking model."
                ) from exc

        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return candidate passages ordered by cross-encoder relevance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            raise ValueError("Candidates must be a sequence of passage mappings.")

        if not candidates:
            return []

        passage_copies: list[dict[str, Any]] = []

        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("Every candidate must be a passage mapping.")

            text = candidate.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every candidate must contain non-empty text.")

            passage_copies.append(dict(candidate))

        try:
            from flashrank import RerankRequest

            ranker = self._get_ranker()
            results = ranker.rerank(
                RerankRequest(
                    query=query,
                    passages=passage_copies,
                )
            )
        except Exception as exc:
            raise RuntimeError("Failed to rerank document passages.") from exc

        if not isinstance(results, list):
            raise RuntimeError("FlashRank returned an invalid result collection.")

        validated_results: list[dict[str, Any]] = []

        for result in results:
            if not isinstance(result, Mapping):
                raise RuntimeError("FlashRank returned an invalid passage.")

            score = result.get("score")

            if (
                isinstance(score, bool)
                or not isinstance(score, numbers.Real)
                or not math.isfinite(score)
            ):
                raise RuntimeError("FlashRank returned an invalid relevance score.")

            validated_results.append(
                {
                    **result,
                    "score": float(score),
                }
            )

        if len(validated_results) != len(candidates):
            raise RuntimeError(
                "FlashRank returned a different number of passages "
                "than it received."
            )

        return validated_results

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured FlashRank model tokenizer."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")

        try:
            return len(self._get_ranker().tokenizer.encode(text).ids)
        except Exception as exc:
            raise RuntimeError("Failed to count passage tokens.") from exc
