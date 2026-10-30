
"""Tests for Groq language model client initialization."""

from unittest.mock import MagicMock

from pdf_rag.llm_client import create_groq_client
from pdf_rag.llm_config import GroqConfig


def test_create_groq_client_uses_configured_settings(monkeypatch) -> None:
    import pdf_rag.llm_client as client_module

    fake_client = MagicMock()
    captured: dict = {}

    def fake_chat_groq(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(client_module, "ChatGroq", fake_chat_groq)

    config = GroqConfig(
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
        timeout_seconds=45.0,
        max_retries=0,
        temperature=0.2,
        max_output_tokens=512,
    )

    result = create_groq_client(config)

    assert result is fake_client
    assert captured == {
        "model": "openai/gpt-oss-20b",
        "api_key": "test-api-key",
        "temperature": 0.2,
        "max_tokens": 512,
        "timeout": 45.0,
        "max_retries": 0,
    }
