"""Tests for shared PDF input validation."""

from io import BytesIO

import pytest

from pdf_rag.ingestion_exceptions import InvalidPDFError, ResourceLimitError
from pdf_rag.validation import (
    DEFAULT_LIMITS,
    IngestionLimits,
    get_source_name,
    validate_pdf_input,
)


VALID_HEADER = b"%PDF-1.7"


def test_default_limits_match_approved_configuration():
    assert DEFAULT_LIMITS.max_file_size_bytes == 25 * 1024 * 1024
    assert DEFAULT_LIMITS.max_pages == 500
    assert DEFAULT_LIMITS.max_extracted_text_bytes == 10 * 1024 * 1024
    assert DEFAULT_LIMITS.chunk_size == 1000
    assert DEFAULT_LIMITS.chunk_overlap == 150


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_file_size_bytes": 0},
        {"max_pages": 0},
        {"max_extracted_text_bytes": 0},
        {"chunk_size": 0},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 101},
        {"chunk_size": 100, "chunk_overlap": -1},
    ],
)
def test_limits_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        IngestionLimits(**kwargs)


def test_validate_pdf_input_accepts_valid_header_and_restores_position():
    stream = BytesIO(VALID_HEADER + b" synthetic PDF content")
    stream.name = "sample.pdf"
    stream.seek(3)

    assert validate_pdf_input(stream) == "sample.pdf"
    assert stream.tell() == 3


@pytest.mark.parametrize("filename", ["sample.PDF", "report.pdf"])
def test_validate_pdf_input_accepts_pdf_extension(filename):
    stream = BytesIO(VALID_HEADER + b" content")
    stream.name = filename

    assert validate_pdf_input(stream) == filename


def test_validate_pdf_input_rejects_non_pdf_extension():
    stream = BytesIO(VALID_HEADER + b" content")
    stream.name = "sample.txt"

    with pytest.raises(InvalidPDFError, match="extension"):
        validate_pdf_input(stream)


def test_validate_pdf_input_rejects_invalid_signature():
    stream = BytesIO(b"NOT A PDF")
    stream.name = "sample.pdf"

    with pytest.raises(InvalidPDFError, match="signature"):
        validate_pdf_input(stream)


def test_validate_pdf_input_rejects_oversized_stream():
    limits = IngestionLimits(max_file_size_bytes=8)
    stream = BytesIO(VALID_HEADER + b" too large")
    stream.name = "sample.pdf"

    with pytest.raises(ResourceLimitError, match="file-size limit"):
        validate_pdf_input(stream, limits)


def test_validate_pdf_input_rejects_unnamed_stream():
    stream = BytesIO(VALID_HEADER + b" content")

    with pytest.raises(InvalidPDFError, match="filename"):
        validate_pdf_input(stream)


def test_get_source_name_returns_basename_for_windows_path():
    assert get_source_name(r"C:\private\documents\sample.pdf") == "sample.pdf"


def test_validate_pdf_input_rejects_missing_path(tmp_path):
    path = tmp_path / "missing.pdf"

    with pytest.raises(InvalidPDFError, match="does not exist"):
        validate_pdf_input(path)
