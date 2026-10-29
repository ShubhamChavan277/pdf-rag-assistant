
"""Text chunking with source-page provenance for extracted PDF pages."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_rag.ingestion_exceptions import (
    NoExtractableTextError,
    PDFExtractionError,
    ResourceLimitError,
)
from pdf_rag.ingestion_models import ChunkRecord, PageRecord
from pdf_rag.validation import DEFAULT_LIMITS, IngestionLimits


def chunk_pages(
    pages: list[PageRecord],
    limits: IngestionLimits = DEFAULT_LIMITS,
) -> list[ChunkRecord]:
    """Split extracted page text into chunks while preserving page provenance.

    Pages must be supplied in ascending, one-based order and must belong to
    the same document and source. A chunk's page_numbers contains every
    source page whose text overlaps the chunk's character range.
    """
    if not pages:
        raise NoExtractableTextError(
            "No extracted pages were provided for chunking."
        )

    first_page = pages[0]

    for expected_number, page in enumerate(pages, start=1):
        if page.document_id != first_page.document_id:
            raise PDFExtractionError(
                "All pages must belong to the same document."
            )

        if page.source_name != first_page.source_name:
            raise PDFExtractionError(
                "All pages must have the same source filename."
            )

        if page.parser != first_page.parser:
            raise PDFExtractionError(
                "All pages must use the same PDF parser."
            )

        if page.page_number != expected_number:
            raise PDFExtractionError(
                "Pages must be consecutive and in ascending order "
                "starting from page 1."
            )

    extracted_text_bytes = sum(
        len(page.text.encode("utf-8")) for page in pages
    )

    if extracted_text_bytes > limits.max_extracted_text_bytes:
        raise ResourceLimitError(
            "The PDF exceeds the configured extracted-text limit."
        )

    # Keep each page's text range in the combined document. The separators
    # preserve a boundary between pages without attributing text to a page.
    combined_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        if combined_parts:
            combined_parts.append("\n\n")
            cursor += 2

        start = cursor
        combined_parts.append(page.text)
        cursor += len(page.text)
        end = cursor

        page_ranges.append((page.page_number, start, end))

    combined_text = "".join(combined_parts)

    if not combined_text.strip():
        raise NoExtractableTextError(
            "No usable text was found to create chunks."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=limits.chunk_size,
        chunk_overlap=limits.chunk_overlap,
        add_start_index=True,
    )

    split_documents = splitter.create_documents([combined_text])
    chunks: list[ChunkRecord] = []

    for chunk_index, document in enumerate(split_documents):
        chunk_text = document.page_content
        start_index = document.metadata.get("start_index")

        if not isinstance(start_index, int) or start_index < 0:
            raise PDFExtractionError(
                "The text splitter did not return a valid chunk position."
            )

        end_index = start_index + len(chunk_text)

        # Include every page whose text range overlaps this chunk. Empty
        # pages and page separators do not independently claim provenance.
        page_numbers = tuple(
            page_number
            for page_number, page_start, page_end in page_ranges
            if page_start < end_index and page_end > start_index
        )

        if not page_numbers:
            raise PDFExtractionError(
                "Could not determine the source pages for a text chunk."
            )

        chunks.append(
            ChunkRecord(
                document_id=first_page.document_id,
                source_name=first_page.source_name,
                chunk_index=chunk_index,
                text=chunk_text,
                page_numbers=page_numbers,
            )
        )

    return chunks
