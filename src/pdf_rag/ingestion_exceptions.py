"""Controlled errors for PDF ingestion and chunking."""


class IngestionError(Exception):
    """Base class for expected PDF ingestion failures."""


class InvalidPDFError(IngestionError):
    """Raised when an input is not a valid supported PDF input."""


class EncryptedPDFError(IngestionError):
    """Raised when a PDF requires a password that was not provided."""


class PDFExtractionError(IngestionError):
    """Raised when a PDF cannot be extracted reliably."""


class NoExtractableTextError(IngestionError):
    """Raised when a document contains no usable extractable text."""


class ResourceLimitError(IngestionError):
    """Raised when a configured ingestion or chunking limit is exceeded."""
