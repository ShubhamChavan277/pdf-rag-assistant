
"""Configuration for the Groq-hosted language model."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

from pdf_rag.exceptions import LLMConfigurationError


@dataclass(frozen=True)
class GroqConfig:
    """Validated configuration for Groq API access."""

    api_key: str = field(repr=False)
    model: str = "openai/gpt-oss-20b"
    timeout_seconds: float = 30.0
    max_retries: int = 0
    temperature: float = 0.0
    max_output_tokens: int = 512


def load_groq_config(
    environ: Mapping[str, str] | None = None,
) -> GroqConfig:
    """Load and validate Groq configuration from environment variables."""

    if environ is None:
        load_dotenv()
        environ = os.environ

    raw_api_key = environ.get("GROQ_API_KEY", "")
    if not isinstance(raw_api_key, str):
        raise LLMConfigurationError("GROQ_API_KEY must be a string.")

    api_key = raw_api_key.strip()
    if not api_key:
        raise LLMConfigurationError(
            "GROQ_API_KEY is missing. Configure it in your local .env "
            "file or deployment secrets."
        )

    raw_model = environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    if not isinstance(raw_model, str):
        raise LLMConfigurationError("GROQ_MODEL must be a string.")

    model = raw_model.strip()

    if not model:
        raise LLMConfigurationError("GROQ_MODEL cannot be empty.")

    try:
        timeout_seconds = float(environ.get("GROQ_TIMEOUT_SECONDS", "30"))
        max_retries = int(environ.get("GROQ_MAX_RETRIES", "0"))
        temperature = float(environ.get("GROQ_TEMPERATURE", "0"))
        max_output_tokens = int(
            environ.get("GROQ_MAX_OUTPUT_TOKENS", "512")
        )
    except ValueError as exc:
        raise LLMConfigurationError(
            "One or more Groq configuration values have an invalid format."
        ) from exc

    if not math.isfinite(timeout_seconds):
        raise LLMConfigurationError(
            "GROQ_TIMEOUT_SECONDS must be a finite number."
        )

    if not math.isfinite(temperature):
        raise LLMConfigurationError(
            "GROQ_TEMPERATURE must be a finite number."
        )

    if timeout_seconds <= 0:
        raise LLMConfigurationError(
            "GROQ_TIMEOUT_SECONDS must be greater than zero."
        )

    if max_retries < 0:
        raise LLMConfigurationError(
            "GROQ_MAX_RETRIES cannot be negative."
        )

    if not 0 <= temperature <= 2:
        raise LLMConfigurationError(
            "GROQ_TEMPERATURE must be between 0 and 2."
        )

    if max_output_tokens <= 0:
        raise LLMConfigurationError(
            "GROQ_MAX_OUTPUT_TOKENS must be greater than zero."
        )

    return GroqConfig(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
