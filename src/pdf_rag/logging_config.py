
"""Logging configuration for PDF RAG Assistant."""

import logging

from pdf_rag.config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
