"""Tests for PDF ingestion and chunk data models."""

import pytest

from pdf_rag.ingestion_models import ChunkRecord, PageRecord


def test_page_record_accepts_valid_metadata():
    page = PageRecord(
        document_id="doc-1",
        source_name="sample.pdf",
        page_number=1,
        text="Extracted page text",
        parser="pypdf",
    )

    assert page.document_id == "doc-1"
    assert page.page_number == 1
    assert page.parser == "pypdf"


@pytest.mark.parametrize(
    "overrides",
    [
        {"document_id": ""},
        {"source_name": ""},
        {"page_number": 0},
        {"page_number": -1},
        {"parser": "unknown"},
    ],
)
def test_page_record_rejects_invalid_metadata(overrides):
    values = {
        "document_id": "doc-1",
        "source_name": "sample.pdf",
        "page_number": 1,
        "text": "Extracted page text",
        "parser": "pypdf",
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        PageRecord(**values)


def test_chunk_record_accepts_valid_metadata():
    chunk = ChunkRecord(
        document_id="doc-1",
        source_name="sample.pdf",
        chunk_index=0,
        text="Extracted chunk text",
        page_numbers=(1, 2),
    )

    assert chunk.chunk_index == 0
    assert chunk.page_numbers == (1, 2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"document_id": ""},
        {"source_name": ""},
        {"chunk_index": -1},
        {"text": ""},
        {"text": "   "},
        {"page_numbers": ()},
        {"page_numbers": (0,)},
        {"page_numbers": (2, 1)},
        {"page_numbers": (1, 1)},
    ],
)
def test_chunk_record_rejects_invalid_metadata(overrides):
    values = {
        "document_id": "doc-1",
        "source_name": "sample.pdf",
        "chunk_index": 0,
        "text": "Extracted chunk text",
        "page_numbers": (1,),
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        ChunkRecord(**values)


def test_records_are_immutable():
    page = PageRecord("doc-1", "sample.pdf", 1, "Text", "pypdf")
    chunk = ChunkRecord("doc-1", "sample.pdf", 0, "Text", (1,))

    with pytest.raises(AttributeError):
        page.page_number = 2

    with pytest.raises(AttributeError):
        chunk.chunk_index = 1
