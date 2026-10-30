
"""Tests for Groq language model configuration."""

import pytest

from pdf_rag.exceptions import LLMConfigurationError
from pdf_rag.llm_config import GroqConfig, load_groq_config


def test_load_groq_config_uses_defaults() -> None:
    config = load_groq_config({"GROQ_API_KEY": "test-key"})

    assert isinstance(config, GroqConfig)
    assert config.model == "openai/gpt-oss-20b"
    assert config.timeout_seconds == 30.0
    assert config.max_retries == 0
    assert config.temperature == 0.0
    assert config.max_output_tokens == 512


def test_load_groq_config_reads_environment_overrides() -> None:
    config = load_groq_config(
        {
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "custom-model",
            "GROQ_TIMEOUT_SECONDS": "45",
            "GROQ_MAX_RETRIES": "2",
            "GROQ_TEMPERATURE": "0.5",
            "GROQ_MAX_OUTPUT_TOKENS": "1024",
        }
    )

    assert config.model == "custom-model"
    assert config.timeout_seconds == 45.0
    assert config.max_retries == 2
    assert config.temperature == 0.5
    assert config.max_output_tokens == 1024


@pytest.mark.parametrize("api_key", ["", " ", None])
def test_load_groq_config_rejects_missing_api_key(
    api_key: str | None,
) -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        load_groq_config({"GROQ_API_KEY": api_key})


def test_load_groq_config_rejects_empty_model() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_MODEL"):
        load_groq_config(
            {
                "GROQ_API_KEY": "test-key",
                "GROQ_MODEL": " ",
            }
        )


def test_load_groq_config_rejects_non_string_model() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_MODEL"):
        load_groq_config(
            {
                "GROQ_API_KEY": "test-key",
                "GROQ_MODEL": None,
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GROQ_TIMEOUT_SECONDS", "invalid"),
        ("GROQ_MAX_RETRIES", "invalid"),
        ("GROQ_TEMPERATURE", "invalid"),
        ("GROQ_MAX_OUTPUT_TOKENS", "invalid"),
    ],
)
def test_load_groq_config_rejects_invalid_numeric_values(
    name: str,
    value: str,
) -> None:
    with pytest.raises(LLMConfigurationError, match="invalid format"):
        load_groq_config(
            {
                "GROQ_API_KEY": "test-key",
                name: value,
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GROQ_TIMEOUT_SECONDS", "nan"),
        ("GROQ_TIMEOUT_SECONDS", "inf"),
        ("GROQ_TEMPERATURE", "nan"),
        ("GROQ_TEMPERATURE", "inf"),
    ],
)
def test_load_groq_config_rejects_non_finite_values(
    name: str,
    value: str,
) -> None:
    with pytest.raises(LLMConfigurationError, match="finite number"):
        load_groq_config(
            {
                "GROQ_API_KEY": "test-key",
                name: value,
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GROQ_TIMEOUT_SECONDS", "0"),
        ("GROQ_TIMEOUT_SECONDS", "-1"),
        ("GROQ_MAX_RETRIES", "-1"),
        ("GROQ_TEMPERATURE", "-0.1"),
        ("GROQ_TEMPERATURE", "2.1"),
        ("GROQ_MAX_OUTPUT_TOKENS", "0"),
        ("GROQ_MAX_OUTPUT_TOKENS", "-1"),
    ],
)
def test_load_groq_config_rejects_out_of_range_values(
    name: str,
    value: str,
) -> None:
    with pytest.raises(LLMConfigurationError):
        load_groq_config(
            {
                "GROQ_API_KEY": "test-key",
                name: value,
            }
        )


def test_groq_api_key_is_not_exposed_in_config_repr() -> None:
    config = load_groq_config({"GROQ_API_KEY": "secret-test-key"})

    assert "secret-test-key" not in repr(config)
