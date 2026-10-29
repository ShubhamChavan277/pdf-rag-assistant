"""PDF text extraction adapter implemented with pdfplumber."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import pdfplumber
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from pdf_rag.ingestion_exceptions import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    PDFExtractionError,
    ResourceLimitError,
)
from pdf_rag.ingestion_models import PageRecord
from pdf_rag.validation import DEFAULT_LIMITS, IngestionLimits, validate_pdf_input



def _read_source_bytes(
    source: str | Path | BinaryIO,
    max_bytes: int,
) -> bytes:
    """Read at most max_bytes + 1 bytes while preserving stream position."""
    if isinstance(source, (str, Path)):
        try:
            with Path(source).open("rb") as stream:
                content = stream.read(max_bytes + 1)
        except OSError as exc:
            raise InvalidPDFError("The PDF file could not be read.") from exc
    else:
        original_position = None

        try:
            original_position = source.tell()
            source.seek(0)
            content = source.read(max_bytes + 1)
        except (OSError, ValueError) as exc:
            raise InvalidPDFError("The PDF stream could not be read.") from exc
        finally:
            if original_position is not None:
                try:
                    source.seek(original_position)
                except (OSError, ValueError):
                    pass

    if not isinstance(content, bytes):
        raise InvalidPDFError("The PDF input must provide binary data.")

    if len(content) > max_bytes:
        raise ResourceLimitError(
            "The PDF exceeds the configured file-size limit."
        )

    return content


def extract_pages_pdfplumber(
    source: str | Path | BinaryIO,
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> list[PageRecord]:
    """Extract page text and metadata from a PDF using pdfplumber."""
    source_name = validate_pdf_input(source, limits)
    pdf_bytes = _read_source_bytes(source, limits.max_file_size_bytes)

    # Use pypdf for a consistent preflight check for encryption and page count.
    try:
        preflight_reader = PdfReader(BytesIO(pdf_bytes), strict=True)

        if preflight_reader.is_encrypted:
            try:
                decrypted = preflight_reader.decrypt("")
            except Exception as exc:
                raise EncryptedPDFError(
                    "The PDF is encrypted and requires a password."
                ) from exc

            if not decrypted:
                raise EncryptedPDFError(
                    "The PDF is encrypted and requires a password."
                )

        page_count = len(preflight_reader.pages)

        if page_count > limits.max_pages:
            raise ResourceLimitError(
                "The PDF exceeds the configured page-count limit."
            )

    except (
        EncryptedPDFError,
        ResourceLimitError,
    ):
        raise
    except FileNotDecryptedError as exc:
        raise EncryptedPDFError(
            "The PDF is encrypted and requires a password."
        ) from exc
    except (PdfReadError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise InvalidPDFError(
            "The PDF could not be parsed as a valid, supported document."
        ) from exc
    except Exception as exc:
        raise PDFExtractionError(
            "The PDF could not be inspected before extraction."
        ) from exc

    document_id = sha256(pdf_bytes).hexdigest()[:16] + "-" + uuid4().hex[:8]
    pages: list[PageRecord] = []
    extracted_text_bytes = 0
    has_usable_text = False

    try:
        with pdfplumber.open(BytesIO(pdf_bytes), repair=False) as pdf:
            if len(pdf.pages) != page_count:
                raise PDFExtractionError(
                    "The PDF page count differed between parser checks."
                )

            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    raise PDFExtractionError(
                        f"Text extraction failed on page {page_number}."
                    ) from exc

                extracted_text_bytes += len(text.encode("utf-8"))

                if extracted_text_bytes > limits.max_extracted_text_bytes:
                    raise ResourceLimitError(
                        "The PDF exceeds the configured extracted-text limit."
                    )

                if text.strip():
                    has_usable_text = True

                pages.append(
                    PageRecord(
                        document_id=document_id,
                        source_name=source_name,
                        page_number=page_number,
                        text=text,
                        parser="pdfplumber",
                    )
                )

    except (
        EncryptedPDFError,
        NoExtractableTextError,
        PDFExtractionError,
        ResourceLimitError,
    ):
        raise
    except Exception as exc:
        raise PDFExtractionError(
            "The PDF could not be opened or extracted by pdfplumber."
        ) from exc

    if not has_usable_text:
        raise NoExtractableTextError(
            "No extractable text was found in this PDF. "
            "It may contain scanned images; OCR is not enabled."
        )

    return pages
