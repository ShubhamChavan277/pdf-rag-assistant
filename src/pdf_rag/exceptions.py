
"""Application-specific exceptions for PDF RAG Assistant."""


class PDFRAGError(Exception):
    """Base exception for expected application-level failures."""


class DependencyError(PDFRAGError):
    """Raised when a required application dependency is unavailable."""
