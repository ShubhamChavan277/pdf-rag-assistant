
"""Groq language model client initialization."""

from langchain_groq import ChatGroq

from pdf_rag.exceptions import DependencyError
from pdf_rag.llm_config import GroqConfig


def create_groq_client(config: GroqConfig) -> ChatGroq:
    """Create a configured LangChain ChatGroq client."""

    try:
        return ChatGroq(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    except ImportError as exc:
        raise DependencyError(
            "The langchain-groq package is required for Groq integration."
        ) from exc
