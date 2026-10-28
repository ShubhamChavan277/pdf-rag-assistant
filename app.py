
"""Streamlit entry point for PDF RAG Assistant."""

import logging

import streamlit as st

from pdf_rag.config import ConfigurationError, load_config
from pdf_rag.dependencies import check_dependencies
from pdf_rag.exceptions import PDFRAGError
from pdf_rag.logging_config import configure_logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Render the application foundation."""
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
        "Ask questions about your PDF documents using retrieval-augmented "
        "generation. Document processing and question answering will be "
        "introduced in later development milestones."
    )

    st.info("Application foundation initialized. No documents have been processed.")


if __name__ == "__main__":
    main()
