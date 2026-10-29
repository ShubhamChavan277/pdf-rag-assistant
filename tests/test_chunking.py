
"""Tests for PDF text chunking and source-page provenance."""

import pytest

from pdf_rag.chunking import chunk_pages
from pdf_rag.ingestion_exceptions import (
    NoExtractableTextError,
    PDFExtractionError,
    ResourceLimitError,
)
from pdf_rag.ingestion_models import PageRecord
from pdf_rag.validation import IngestionLimits


def make_page(
    text: str,
    page_number: int = 1,
    *,
    document_id: str = "document-123",
    source_name: str = "sample.pdf",
    parser: str = "pypdf",
) -> PageRecord:
    """Build a page record for chunking tests."""
    return PageRecord(
        document_id=document_id,
        source_name=source_name,
        page_number=page_number,
        text=text,
        parser=parser,
    )


def test_chunk_pages_returns_text_and_source_metadata():
    pages = [make_page("Hello PDF")]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello PDF"
    assert chunks[0].document_id == "document-123"
    assert chunks[0].source_name == "sample.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == (1,)


def test_chunk_indexes_are_sequential():
    pages = [make_page("abcdefghijklmnopqrstuvwxyz")]
    limits = IngestionLimits(chunk_size=10, chunk_overlap=2)

    chunks = chunk_pages(pages, limits)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )


def test_chunks_respect_configured_size():
    pages = [make_page("abcdefghijklmnopqrstuvwxyz")]
    limits = IngestionLimits(chunk_size=10, chunk_overlap=2)

    chunks = chunk_pages(pages, limits)

    assert all(len(chunk.text) <= 10 for chunk in chunks)


def test_adjacent_chunks_preserve_overlap():
    pages = [make_page("abcdefghijklmnopqrstuvwxyz")]
    limits = IngestionLimits(chunk_size=10, chunk_overlap=3)

    chunks = chunk_pages(pages, limits)

    assert len(chunks) > 1
    assert chunks[0].text[-3:] == chunks[1].text[:3]


def test_chunk_spanning_two_pages_records_both_pages():
    pages = [
        make_page("First page.", 1),
        make_page("Second page.", 2),
    ]
    limits = IngestionLimits(chunk_size=100, chunk_overlap=10)

    chunks = chunk_pages(pages, limits)

    assert len(chunks) == 1
    assert chunks[0].page_numbers == (1, 2)
    assert "First page." in chunks[0].text
    assert "Second page." in chunks[0].text


def test_chunk_metadata_has_unique_sorted_page_numbers():
    pages = [
        make_page("First page text.", 1),
        make_page("Second page text.", 2),
        make_page("Third page text.", 3),
    ]
    limits = IngestionLimits(chunk_size=20, chunk_overlap=5)

    chunks = chunk_pages(pages, limits)

    for chunk in chunks:
        assert chunk.page_numbers
        assert tuple(sorted(set(chunk.page_numbers))) == chunk.page_numbers


def test_chunk_pages_rejects_empty_page_list():
    with pytest.raises(NoExtractableTextError, match="No extracted pages"):
        chunk_pages([])


def test_chunk_pages_rejects_pages_without_usable_text():
    pages = [
        make_page("   ", 1),
        make_page("\n\t", 2),
    ]

    with pytest.raises(NoExtractableTextError, match="No usable text"):
        chunk_pages(pages)


def test_chunk_pages_rejects_nonconsecutive_page_numbers():
    pages = [
        make_page("Page one.", 1),
        make_page("Page three.", 3),
    ]

    with pytest.raises(PDFExtractionError, match="consecutive"):
        chunk_pages(pages)


def test_chunk_pages_rejects_mixed_document_ids():
    pages = [
        make_page("Page one.", 1, document_id="document-a"),
        make_page("Page two.", 2, document_id="document-b"),
    ]

    with pytest.raises(PDFExtractionError, match="same document"):
        chunk_pages(pages)


def test_chunk_pages_rejects_mixed_source_names():
    pages = [
        make_page("Page one.", 1, source_name="first.pdf"),
        make_page("Page two.", 2, source_name="second.pdf"),
    ]

    with pytest.raises(PDFExtractionError, match="same source filename"):
        chunk_pages(pages)


def test_chunk_pages_rejects_mixed_parsers():
    pages = [
        make_page("Page one.", 1, parser="pypdf"),
        make_page("Page two.", 2, parser="pdfplumber"),
    ]

    with pytest.raises(PDFExtractionError, match="same PDF parser"):
        chunk_pages(pages)


def test_chunk_pages_enforces_extracted_text_limit():
    pages = [make_page("This text exceeds the configured limit.")]
    limits = IngestionLimits(max_extracted_text_bytes=5)

    with pytest.raises(ResourceLimitError, match="extracted-text limit"):
        chunk_pages(pages, limits)
