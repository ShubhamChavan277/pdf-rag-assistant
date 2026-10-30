
"""End-to-end tests for the retrieval-to-answer pipeline."""

from collections.abc import Sequence
from typing import Any

from pdf_rag.answer_generation import ABSTENTION_MESSAGE, AnswerGenerator
from pdf_rag.exceptions import LLMProviderError
from pdf_rag.indexing import index_document
from pdf_rag.ingestion_models import ChunkRecord
from pdf_rag.llm_config import GroqConfig
from pdf_rag.reranking import PassageReranker
from pdf_rag.retrieval import retrieve_chunks


class FakeEmbeddingModel:
    """Return deterministic embeddings without loading a real model."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeRanker:
    """Rank passages deterministically using their content."""

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        passages = request.passages

        for passage in passages:
            passage["score"] = (
                0.99 if "retrieval pipeline" in passage["text"].lower()
                else 0.10
            )

        return sorted(
            passages,
            key=lambda passage: passage["score"],
            reverse=True,
        )


class FakeGroqClient:
    """Capture prompt messages and return a deterministic answer."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[Any] | None = None
        self.call_count = 0

    def invoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        self.call_count += 1
        return type("FakeResponse", (), {"content": self.content})()


def make_chunks() -> list[ChunkRecord]:
    """Build document chunks with distinct source and page metadata."""

    return [
        ChunkRecord(
            document_id="doc-1",
            source_name="retrieval-guide.pdf",
            chunk_index=0,
            text=(
                "A retrieval pipeline finds relevant document passages "
                "using semantic similarity."
            ),
            page_numbers=(2,),
        ),
        ChunkRecord(
            document_id="doc-1",
            source_name="retrieval-guide.pdf",
            chunk_index=1,
            text="Unrelated information about an unrelated topic.",
            page_numbers=(5,),
        ),
    ]


def make_generator(client: FakeGroqClient) -> AnswerGenerator:
    """Build an answer generator using a fake Groq client."""

    config = GroqConfig(api_key="test-api-key")
    return AnswerGenerator(config=config, client=client)


def test_end_to_end_pipeline_generates_grounded_answer() -> None:
    """Exercise indexing, retrieval, reranking, selection, and generation."""

    session_state: dict[str, object] = {}
    embedding_model = FakeEmbeddingModel()

    index_document(make_chunks(), session_state, embedding_model)

    retrieved = retrieve_chunks(
        "Explain the retrieval pipeline",
        session_state,
        embedding_model,
    )

    assert len(retrieved) == 2

    reranker = PassageReranker(ranker=FakeRanker())
    reranked = reranker.rerank(
        "Explain the retrieval pipeline",
        retrieved,
    )

    assert "retrieval pipeline" in reranked[0]["text"].lower()
    assert reranked[0]["score"] > reranked[1]["score"]

    client = FakeGroqClient(
        "A retrieval pipeline finds relevant document passages "
        "using semantic similarity. [retrieval-guide.pdf, page 2]"
    )
    generator = make_generator(client)

    result = generator.generate_answer(
        "Explain the retrieval pipeline",
        reranked,
    )

    assert result.answer.startswith("A retrieval pipeline")
    assert len(result.sources) == 2
    assert result.sources[0]["source_name"] == "retrieval-guide.pdf"
    assert result.sources[0]["page_numbers"] == (2,)
    assert result.sources[0]["score"] == 0.99
    assert client.call_count == 1
    assert client.messages is not None

    prompt_content = "\n".join(
        str(message.content) for message in client.messages
    )
    assert "Explain the retrieval pipeline" in prompt_content
    assert "semantic similarity" in prompt_content
    assert "retrieval-guide.pdf" in prompt_content


def test_end_to_end_pipeline_abstains_when_no_documents_are_indexed() -> None:
    """Verify empty retrieval abstains without invoking the LLM."""

    session_state: dict[str, object] = {}
    embedding_model = FakeEmbeddingModel()

    # An empty session has no active index, so construct an empty index.
    from pdf_rag.indexing import DocumentIndex, SESSION_INDEX_KEY
    from pdf_rag.vector_store import VectorStore

    session_state[SESSION_INDEX_KEY] = DocumentIndex(
        fingerprint="empty",
        store=VectorStore(),
        chunk_count=0,
    )

    retrieved = retrieve_chunks(
        "Question without indexed documents",
        session_state,
        embedding_model,
    )

    reranked = PassageReranker(ranker=FakeRanker()).rerank(
        "Question without indexed documents",
        retrieved,
    )

    client = FakeGroqClient("This response must not be generated.")
    generator = make_generator(client)

    result = generator.generate_answer(
        "Question without indexed documents",
        reranked,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert result.sources == ()
    assert client.call_count == 0
