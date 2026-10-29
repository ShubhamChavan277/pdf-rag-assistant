"""Shared data models for PDF ingestion and chunking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageRecord:
    """Extracted text and source metadata for one PDF page."""

    document_id: str
    source_name: str
    page_number: int
    text: str
    parser: str

    def __post_init__(self) -> None:
        """Validate page metadata."""
        if not self.document_id.strip():
            raise ValueError("Document ID must not be empty.")
        if not self.source_name.strip():
            raise ValueError("Source name must not be empty.")
        if self.page_number < 1:
            raise ValueError("Page numbers must be one-based positive integers.")
        if self.parser not in {"pypdf", "pdfplumber"}:
            raise ValueError("Parser must be 'pypdf' or 'pdfplumber'.")


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk with document identity and page-level provenance."""

    document_id: str
    source_name: str
    chunk_index: int
    text: str
    page_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate chunk metadata and source attribution."""
        if not self.document_id.strip():
            raise ValueError("Document ID must not be empty.")
        if not self.source_name.strip():
            raise ValueError("Source name must not be empty.")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative.")
        if not self.text.strip():
            raise ValueError("Chunk text must not be empty.")
        if not self.page_numbers:
            raise ValueError("A chunk must reference at least one page.")
        if any(page_number < 1 for page_number in self.page_numbers):
            raise ValueError("Page numbers must be positive integers.")
        if tuple(sorted(set(self.page_numbers))) != self.page_numbers:
            raise ValueError(
                "Chunk page numbers must be unique and in ascending order."
            )
