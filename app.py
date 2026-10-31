
"""Streamlit application for PDF RAG Assistant."""

import hashlib
import logging

import streamlit as st

from pdf_rag.answer_generation import AnswerGenerator
from pdf_rag.chunking import chunk_pages
from pdf_rag.config import ConfigurationError, load_config
from pdf_rag.dependencies import check_dependencies
from pdf_rag.embeddings import EmbeddingModel
from pdf_rag.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    PDFRAGError,
)
from pdf_rag.indexing import (
    SESSION_INDEX_KEY,
    DocumentIndex,
    index_document,
)
from pdf_rag.llm_config import load_groq_config
from pdf_rag.logging_config import configure_logging
from pdf_rag.pdfplumber_adapter import extract_pages_pdfplumber
from pdf_rag.pypdf_adapter import extract_pages_pypdf
from pdf_rag.reranking import PassageReranker
from pdf_rag.retrieval import retrieve_chunks
from pdf_rag.ingestion_exceptions import IngestionError


logger = logging.getLogger(__name__)

UPLOAD_FINGERPRINT_KEY = "pdf_rag_upload_fingerprint"
UPLOAD_FILENAME_KEY = "pdf_rag_upload_filename"
UPLOAD_STATUS_KEY = "pdf_rag_upload_status"
UPLOAD_ERROR_KEY = "pdf_rag_upload_error"
CHAT_HISTORY_KEY = "pdf_rag_chat_history"


@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    """Return a reusable embedding model."""
    return EmbeddingModel()


@st.cache_resource
def get_reranker() -> PassageReranker:
    """Return a reusable passage reranker."""
    return PassageReranker()


def clear_active_document() -> None:
    """Remove the active index and document-specific conversation history."""
    active_index = st.session_state.pop(SESSION_INDEX_KEY, None)

    if isinstance(active_index, DocumentIndex):
        try:
            active_index.store.clear()
        except Exception:
            logger.exception("Failed to clear the previous document index.")

    st.session_state[CHAT_HISTORY_KEY] = []
    st.session_state[UPLOAD_FILENAME_KEY] = None


def process_uploaded_document(uploaded_file, parser: str) -> None:
    """Extract, chunk, and index a newly uploaded PDF."""
    file_bytes = uploaded_file.getvalue()
    fingerprint = hashlib.sha256(
        parser.encode("utf-8") + b":" + file_bytes
    ).hexdigest()

    if fingerprint == st.session_state.get(UPLOAD_FINGERPRINT_KEY):
        return

    # Invalidate the previous document before attempting replacement.
    clear_active_document()

    st.session_state[UPLOAD_FINGERPRINT_KEY] = fingerprint
    st.session_state[UPLOAD_STATUS_KEY] = "processing"
    st.session_state[UPLOAD_ERROR_KEY] = None

    try:
        uploaded_file.seek(0)

        if parser == "pypdf":
            pages = extract_pages_pypdf(uploaded_file)
        elif parser == "pdfplumber":
            pages = extract_pages_pdfplumber(uploaded_file)
        else:
            raise ValueError("Unsupported PDF parser selection.")

        chunks = chunk_pages(pages)

        index = index_document(
            chunks,
            st.session_state,
            embedding_model=get_embedding_model(),
        )

        st.session_state[UPLOAD_FILENAME_KEY] = uploaded_file.name
        st.session_state[UPLOAD_STATUS_KEY] = "ready"

        logger.info(
            "Indexed PDF '%s' using %s: %d pages, %d chunks.",
            uploaded_file.name,
            parser,
            len(pages),
            index.chunk_count,
        )

    except IngestionError as exc:
        logger.warning("PDF ingestion failed: %s", exc)
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state[UPLOAD_ERROR_KEY] = str(exc)

    except PDFRAGError as exc:
        logger.warning("PDF processing failed: %s", exc)
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state[UPLOAD_ERROR_KEY] = str(exc)

    except (ValueError, RuntimeError) as exc:
        logger.exception("PDF indexing failed.")
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state[UPLOAD_ERROR_KEY] = str(exc)

    except Exception:
        logger.exception("Unexpected PDF processing failure.")
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state[UPLOAD_ERROR_KEY] = (
            "An unexpected error occurred while processing the PDF. "
            "Check the application logs."
        )


def render_document_status() -> None:
    """Display the current document processing status."""
    status = st.session_state.get(UPLOAD_STATUS_KEY, "idle")

    if status == "processing":
        st.info("Processing your PDF...")

    elif status == "ready":
        active_index = st.session_state.get(SESSION_INDEX_KEY)

        if isinstance(active_index, DocumentIndex):
            st.success(
                f"**{st.session_state[UPLOAD_FILENAME_KEY]}** is ready. "
                f"Indexed {active_index.chunk_count} text chunks."
            )

    elif status == "error":
        st.error(
            f"PDF processing failed: "
            f"{st.session_state.get(UPLOAD_ERROR_KEY, 'Unknown error')}"
        )


def generate_response(question: str) -> dict:
    """Run retrieval, reranking, and grounded answer generation."""
    groq_config = load_groq_config()

    embedding_model = get_embedding_model()
    reranker = get_reranker()

    candidates = retrieve_chunks(
        question,
        st.session_state,
        embedding_model=embedding_model,
    )

    ranked_passages = reranker.rerank(question, candidates)

    generator = AnswerGenerator(config=groq_config)
    result = generator.generate_answer(question, ranked_passages)

    return {
        "answer": result.answer,
        "sources": list(result.sources),
        "error": None,
    }


def render_sources(sources: list[dict]) -> None:
    """Display the passages selected as supporting evidence."""
    if not sources:
        st.caption("No supporting passages were selected.")
        return

    with st.expander(f"Supporting evidence ({len(sources)} passages)"):
        for index, source in enumerate(sources, start=1):
            pages = ", ".join(
                str(number) for number in source.get("page_numbers", ())
            ) or "Unknown"

            st.markdown(
                f"**Passage {index} — {source.get('source_name', 'Unknown PDF')}**"
            )
            st.caption(f"Page(s): {pages}")
            st.write(source.get("text", ""))
            if index < len(sources):
                st.divider()


def render_chat_history() -> None:
    """Render the current session's question-and-answer history."""
    history = st.session_state.get(CHAT_HISTORY_KEY, [])

    for entry in history:
        with st.chat_message("user"):
            st.write(entry["question"])

        with st.chat_message("assistant"):
            if entry.get("error"):
                st.error(entry["error"])
            else:
                st.write(entry["answer"])
                render_sources(entry.get("sources", []))


def handle_question(question: str) -> None:
    """Generate an answer and append it to the current session history."""
    try:
        response = generate_response(question)

    except LLMConfigurationError as exc:
        response = {
            "answer": "",
            "sources": [],
            "error": (
                f"Groq configuration error: {exc} "
                "Check your .env file and GROQ_API_KEY."
            ),
        }

    except LLMAuthenticationError as exc:
        response = {"answer": "", "sources": [], "error": str(exc)}

    except LLMRateLimitError as exc:
        response = {"answer": "", "sources": [], "error": str(exc)}

    except LLMTimeoutError as exc:
        response = {"answer": "", "sources": [], "error": str(exc)}

    except LLMProviderError as exc:
        response = {"answer": "", "sources": [], "error": str(exc)}

    except (RuntimeError, ValueError) as exc:
        logger.exception("Question processing failed.")
        response = {
            "answer": "",
            "sources": [],
            "error": f"Could not process the question: {exc}",
        }

    except Exception:
        logger.exception("Unexpected question processing failure.")
        response = {
            "answer": "",
            "sources": [],
            "error": (
                "An unexpected error occurred while answering your question. "
                "Check the application logs."
            ),
        }

    history = st.session_state.setdefault(CHAT_HISTORY_KEY, [])
    history.append(
        {
            "question": question,
            **response,
        }
    )


def main() -> None:
    """Render the PDF question-answering application."""
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📄",
        layout="centered",
    )

    try:
        config = load_config()
        configure_logging(config)
        check_dependencies()

    except ConfigurationError as exc:
        st.error(f"Application configuration error: {exc}")
        st.stop()

    except PDFRAGError:
        logger.exception("Application initialization failed.")
        st.error(
            "The application could not initialize because a required "
            "component is unavailable. Check the application logs."
        )
        st.stop()

    st.title(config.app_name)
    st.write(
        "Upload a PDF and ask questions about its contents. "
        "Answers are grounded in the document's extracted text."
    )

    st.subheader("1. Upload your PDF")

    parser = st.selectbox(
        "PDF parser",
        options=("pypdf", "pdfplumber"),
        index=0,
        help="Choose the parser used to extract text from your PDF.",
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Maximum file size: 25 MB.",
    )

    if uploaded_file is None:
        if st.session_state.get(UPLOAD_FINGERPRINT_KEY) is not None:
            clear_active_document()
            st.session_state[UPLOAD_FINGERPRINT_KEY] = None
            st.session_state[UPLOAD_STATUS_KEY] = "idle"
            st.session_state[UPLOAD_ERROR_KEY] = None

    else:
        process_uploaded_document(uploaded_file, parser)

    render_document_status()

    st.divider()
    st.subheader("2. Ask questions")

    active_index = st.session_state.get(SESSION_INDEX_KEY)
    document_ready = (
        isinstance(active_index, DocumentIndex)
        and active_index.store.count() > 0
        and st.session_state.get(UPLOAD_STATUS_KEY) == "ready"
    )

    if document_ready:
        st.caption(
            f"Currently answering questions about: "
            f"{st.session_state.get(UPLOAD_FILENAME_KEY)}"
        )
    else:
        st.info("Upload and successfully process a PDF before asking questions.")

    render_chat_history()

    question = st.chat_input(
        "Ask a question about your PDF...",
        disabled=not document_ready,
        max_chars=2000,
    )

    if question and document_ready:
        with st.spinner("Searching the document and generating an answer..."):
            handle_question(question)

        st.rerun()


if __name__ == "__main__":
    main()
