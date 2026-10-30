
"""Generate evidence-grounded answers using the Groq language model."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from pdf_rag.context_selection import select_context
from pdf_rag.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from pdf_rag.llm_client import create_groq_client
from pdf_rag.llm_config import GroqConfig
from pdf_rag.llm_context import estimate_tokens, format_passages
from pdf_rag.llm_prompts import get_answer_prompt


ABSTENTION_MESSAGE = (
    "I could not find sufficient information in the provided documents "
    "to answer this question."
)

DEFAULT_MAX_CONTEXT_TOKENS = 2000
DEFAULT_MAX_CONTEXT_PASSAGES = 5
MAX_QUESTION_TOKENS = 512


@dataclass(frozen=True)
class AnswerResult:
    """Generated answer and the passages used to support it."""

    answer: str
    sources: tuple[dict[str, Any], ...]


class AnswerGenerator:
    """Generate answers grounded exclusively in selected PDF passages."""

    def __init__(
        self,
        config: GroqConfig,
        client: Any | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_context_passages: int = DEFAULT_MAX_CONTEXT_PASSAGES,
    ) -> None:
        if (
            isinstance(max_context_tokens, bool)
            or not isinstance(max_context_tokens, int)
            or max_context_tokens < 1
        ):
            raise ValueError("Context token budget must be a positive integer.")

        if (
            isinstance(max_context_passages, bool)
            or not isinstance(max_context_passages, int)
            or max_context_passages < 1
        ):
            raise ValueError("Maximum context passages must be a positive integer.")

        self.config = config
        self.client = client if client is not None else create_groq_client(config)
        self.max_context_tokens = max_context_tokens
        self.max_context_passages = max_context_passages

    def generate_answer(
        self,
        question: str,
        ranked_passages: Sequence[Mapping[str, Any]],
    ) -> AnswerResult:
        """Generate an answer from reranked passages or abstain if none fit."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if estimate_tokens(question) > MAX_QUESTION_TOKENS:
            raise ValueError(
                f"Question exceeds the maximum length of "
                f"{MAX_QUESTION_TOKENS} estimated tokens."
            )

        if isinstance(ranked_passages, (str, bytes)) or not isinstance(
            ranked_passages, Sequence
        ):
            raise ValueError("Ranked passages must be a sequence.")

        if not ranked_passages:
            return AnswerResult(answer=ABSTENTION_MESSAGE, sources=())

        selected_passages = select_context(
            ranked_passages,
            token_counter=estimate_tokens,
            max_passages=self.max_context_passages,
            max_context_tokens=self.max_context_tokens,
        )

        if not selected_passages:
            return AnswerResult(answer=ABSTENTION_MESSAGE, sources=())

        context = format_passages(selected_passages)
        prompt = get_answer_prompt()

        try:
            messages = prompt.format_messages(
                question=question.strip(),
                context=context,
            )
            response = self.client.invoke(messages)
        except RateLimitError as exc:
            raise LLMRateLimitError(
                "The Groq API rate limit has been reached. Please try again later."
            ) from exc
        except AuthenticationError as exc:
            raise LLMAuthenticationError(
                "Groq API authentication failed. Check the configured API key."
            ) from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "The Groq API request timed out. Please try again."
            ) from exc
        except APIConnectionError as exc:
            raise LLMProviderError(
                "Could not connect to the Groq API."
            ) from exc
        except APIError as exc:
            raise LLMProviderError(
                "The Groq API returned an error."
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                "Answer generation failed unexpectedly."
            ) from exc

        content = getattr(response, "content", None)

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, Mapping)
                and isinstance(item.get("text"), str)
            ]
            answer = "\n".join(text_parts).strip()
        else:
            answer = ""

        if not answer:
            raise LLMProviderError(
                "The Groq API returned an empty or invalid answer."
            )

        return AnswerResult(
            answer=answer,
            sources=tuple(dict(passage) for passage in selected_passages),
        )
