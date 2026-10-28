
"""Minimal application configuration for PDF RAG Assistant."""

import os
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    """Validated settings required by the application foundation."""

    app_name: str = "PDF RAG Assistant"
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """Load and validate configuration from environment variables."""
    app_name = os.getenv("PDF_RAG_APP_NAME", "PDF RAG Assistant").strip()
    log_level = os.getenv("PDF_RAG_LOG_LEVEL", "INFO").strip().upper()

    if not app_name:
        raise ConfigurationError(
            "PDF_RAG_APP_NAME must not be empty."
        )

    allowed_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in allowed_log_levels:
        allowed = ", ".join(sorted(allowed_log_levels))
        raise ConfigurationError(
            f"Invalid PDF_RAG_LOG_LEVEL. Choose one of: {allowed}."
        )

    return AppConfig(app_name=app_name, log_level=log_level)
