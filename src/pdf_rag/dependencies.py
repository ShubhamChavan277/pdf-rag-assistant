
"""Lightweight checks for required application dependencies."""

from importlib.util import find_spec

from pdf_rag.exceptions import DependencyError


REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "pypdf": "pypdf",
    "pdfplumber": "pdfplumber",
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
    "langchain_groq": "langchain-groq",
    "langchain_text_splitters": "langchain-text-splitters",
    "flashrank": "flashrank",
}


def check_dependencies() -> None:
    """Raise a clear error if a required package cannot be located."""
    missing = [
        package
        for module, package in REQUIRED_PACKAGES.items()
        if find_spec(module) is None
    ]

    if missing:
        packages = ", ".join(missing)
        raise DependencyError(
            f"Required packages are missing or unavailable: {packages}. "
            "Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        )
