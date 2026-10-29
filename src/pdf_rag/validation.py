"""Shared input validation and resource limits for PDF ingestion."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from pdf_rag.ingestion_exceptions import (
    InvalidPDFError,
    ResourceLimitError,
)


MIB = 1024 * 1024


@dataclass(frozen=True)
class IngestionLimits:
    """Configurable resource limits for PDF ingestion and chunking."""

    max_file_size_bytes: int = 25 * MIB
    max_pages: int = 500
    max_extracted_text_bytes: int = 10 * MIB
    chunk_size: int = 1000
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        """Reject invalid or internally inconsistent limits."""
        if self.max_file_size_bytes <= 0:
            raise ValueError("Maximum file size must be positive.")
        if self.max_pages <= 0:
            raise ValueError("Maximum page count must be positive.")
        if self.max_extracted_text_bytes <= 0:
            raise ValueError("Maximum extracted-text size must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative and smaller "
                "than chunk size."
            )


DEFAULT_LIMITS = IngestionLimits()


def get_source_name(source: str | Path | BinaryIO) -> str:
    """Return a safe display filename without directory components."""
    if isinstance(source, (str, Path)):
        name = Path(source).name
    else:
        raw_name = getattr(source, "name", None)
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InvalidPDFError(
                "A filename is required for a file-like PDF input."
            )
        name = PureWindowsPath(raw_name).name
        name = Path(name).name

    if not name or name in {".", ".."}:
        raise InvalidPDFError("The PDF filename is invalid.")

    return name


def validate_pdf_input(
    source: str | Path | BinaryIO,
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> str:
    """Validate the filename, size, and PDF signature without consuming input."""
    name = get_source_name(source)

    if Path(name).suffix.lower() != ".pdf":
        raise InvalidPDFError("The input filename must have a .pdf extension.")

    if isinstance(source, (str, Path)):
        path = Path(source)

        if not path.exists() or not path.is_file():
            raise InvalidPDFError("The PDF file does not exist or is not a file.")

        size = path.stat().st_size

        if size > limits.max_file_size_bytes:
            raise ResourceLimitError(
                "The PDF exceeds the configured file-size limit."
            )

        try:
            with path.open("rb") as stream:
                signature = stream.read(5)
        except OSError as exc:
            raise InvalidPDFError("The PDF file could not be read.") from exc

    else:
        if not all(
            callable(getattr(source, method, None))
            for method in ("read", "seek", "tell")
        ):
            raise InvalidPDFError(
                "The PDF input must be a seekable binary stream."
            )

        try:
            original_position = source.tell()
            source.seek(0, 2)
            size = source.tell()
            source.seek(0)
            signature = source.read(5)
            source.seek(original_position)
        except (OSError, ValueError) as exc:
            raise InvalidPDFError(
                "The PDF stream could not be inspected."
            ) from exc

        if not isinstance(signature, bytes):
            raise InvalidPDFError("The PDF stream must provide binary data.")

        if size > limits.max_file_size_bytes:
            raise ResourceLimitError(
                "The PDF exceeds the configured file-size limit."
            )

    if signature != b"%PDF-":
        raise InvalidPDFError(
            "The file does not have the expected PDF signature."
        )

    return name
