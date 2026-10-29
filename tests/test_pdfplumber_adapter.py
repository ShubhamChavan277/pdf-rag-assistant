
"""Tests for the pdfplumber extraction adapter."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from pdf_rag.ingestion_exceptions import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    ResourceLimitError,
)
from pdf_rag.pdfplumber_adapter import extract_pages_pdfplumber
from pdf_rag.pypdf_adapter import extract_pages_pypdf
from pdf_rag.validation import IngestionLimits


def make_text_pdf(text: str = "Hello PDF") -> bytes:
    """Build a small PDF containing extractable text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )

    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font}
            )
        }
    )
    page[NameObject("/Resources")] = resources

    safe_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    content = (
        f"BT /F1 12 Tf 40 250 Td ({safe_text}) Tj ET"
    ).encode("ascii")

    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def make_blank_pdf(page_count: int) -> bytes:
    """Build a PDF containing the requested number of blank pages."""
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def named_stream(
    content: bytes,
    name: str = "sample.pdf",
) -> BytesIO:
    """Create a named seekable binary stream."""
    stream = BytesIO(content)
    stream.name = name
    return stream


def test_extract_pages_returns_text_and_page_metadata():
    pages = extract_pages_pdfplumber(named_stream(make_text_pdf()))

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].parser == "pdfplumber"
    assert pages[0].source_name == "sample.pdf"
    assert "Hello PDF" in pages[0].text
    assert pages[0].document_id


def test_extract_pages_preserves_stream_position():
    stream = named_stream(make_text_pdf())
    stream.seek(4)

    extract_pages_pdfplumber(stream)

    assert stream.tell() == 4


def test_extract_pages_rejects_invalid_pdf_signature():
    with pytest.raises(InvalidPDFError, match="signature"):
        extract_pages_pdfplumber(named_stream(b"not a PDF"))


def test_extract_pages_rejects_pdf_without_usable_text():
    with pytest.raises(NoExtractableTextError, match="No extractable text"):
        extract_pages_pdfplumber(named_stream(make_blank_pdf(2)))


def test_extract_pages_enforces_page_limit():
    limits = IngestionLimits(max_pages=1)

    with pytest.raises(ResourceLimitError, match="page-count limit"):
        extract_pages_pdfplumber(
            named_stream(make_blank_pdf(2)),
            limits,
        )


def test_extract_pages_enforces_extracted_text_limit():
    limits = IngestionLimits(max_extracted_text_bytes=5)

    with pytest.raises(ResourceLimitError, match="extracted-text limit"):
        extract_pages_pdfplumber(named_stream(make_text_pdf()), limits)


def test_extract_pages_rejects_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("test-password")

    output = BytesIO()
    writer.write(output)

    with pytest.raises(EncryptedPDFError, match="encrypted"):
        extract_pages_pdfplumber(named_stream(output.getvalue()))


def test_extract_pages_rejects_wrong_extension():
    with pytest.raises(InvalidPDFError, match="extension"):
        extract_pages_pdfplumber(
            named_stream(make_text_pdf(), "sample.txt")
        )


def test_both_parsers_extract_text_from_the_same_pdf():
    pdf_bytes = make_text_pdf()

    pypdf_pages = extract_pages_pypdf(named_stream(pdf_bytes))
    pdfplumber_pages = extract_pages_pdfplumber(named_stream(pdf_bytes))

    assert len(pypdf_pages) == len(pdfplumber_pages) == 1
    assert "Hello PDF" in pypdf_pages[0].text
    assert "Hello PDF" in pdfplumber_pages[0].text
    assert pypdf_pages[0].page_number == pdfplumber_pages[0].page_number == 1
    assert pypdf_pages[0].parser == "pypdf"
    assert pdfplumber_pages[0].parser == "pdfplumber"
