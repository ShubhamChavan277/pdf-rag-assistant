
"""Tests for application configuration."""

import pytest

from pdf_rag.config import ConfigurationError, load_config


def test_load_config_uses_defaults(monkeypatch):
    monkeypatch.delenv("PDF_RAG_APP_NAME", raising=False)
    monkeypatch.delenv("PDF_RAG_LOG_LEVEL", raising=False)

    config = load_config()

    assert config.app_name == "PDF RAG Assistant"
    assert config.log_level == "INFO"


def test_load_config_accepts_environment_overrides(monkeypatch):
    monkeypatch.setenv("PDF_RAG_APP_NAME", "Test Assistant")
    monkeypatch.setenv("PDF_RAG_LOG_LEVEL", "debug")

    config = load_config()

    assert config.app_name == "Test Assistant"
    assert config.log_level == "DEBUG"


def test_load_config_rejects_empty_app_name(monkeypatch):
    monkeypatch.setenv("PDF_RAG_APP_NAME", "   ")

    with pytest.raises(ConfigurationError, match="must not be empty"):
        load_config()


def test_load_config_rejects_invalid_log_level(monkeypatch):
    monkeypatch.setenv("PDF_RAG_LOG_LEVEL", "INVALID")

    with pytest.raises(ConfigurationError, match="Invalid PDF_RAG_LOG_LEVEL"):
        load_config()
