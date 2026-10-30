
"""Application-specific exceptions for PDF RAG Assistant."""


class PDFRAGError(Exception):
    """Base exception for expected application-level failures."""


class DependencyError(PDFRAGError):
    """Raised when a required application dependency is unavailable."""


class LLMConfigurationError(PDFRAGError):
    """Raised when LLM provider configuration is missing or invalid."""


class LLMProviderError(PDFRAGError):
    """Raised when the LLM provider returns an unexpected failure."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when LLM provider authentication fails."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider rate limit is exceeded."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""
