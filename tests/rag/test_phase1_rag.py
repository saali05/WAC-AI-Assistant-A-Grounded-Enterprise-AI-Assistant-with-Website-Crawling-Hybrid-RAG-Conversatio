import pytest
from datetime import datetime, UTC
from app.core.config import settings
from app.rag.exceptions import (
    RAGException,
    CrawlerException,
    SSRFProtectionException,
    ExtractionException,
    ChunkingException,
    EmbeddingException,
    RetrievalException,
    GroundingException,
    RAGConfigurationException,
)
from app.rag.models import (
    RAGDocumentModel,
    RAGChunkModel,
    CrawlRunModel,
    SourceCitation,
    RAGResult,
)


def test_rag_configuration():
    """Verify Phase 1 RAG settings are properly defined."""
    assert settings.RAG_ENABLED is True
    assert "webandcrafts.com" in settings.RAG_ALLOWED_DOMAINS
    assert len(settings.allowed_domains_list) >= 2
    assert "webandcrafts.com" in settings.allowed_domains_list
    assert "www.webandcrafts.com" in settings.allowed_domains_list
    assert settings.RAG_EMBEDDING_MODEL == "gemini-embedding-001"
    assert settings.RAG_EMBEDDING_DIMENSIONS == 768
    assert settings.RAG_VECTOR_WEIGHT == 0.7
    assert settings.RAG_KEYWORD_WEIGHT == 0.3
    assert settings.RAG_MIN_RELEVANCE_SCORE == 0.65
    assert settings.RAG_TOP_K_VECTOR == 20
    assert settings.RAG_TOP_K_KEYWORD == 20
    assert settings.RAG_TOP_K_FINAL == 5
    assert settings.RAG_CHUNK_SIZE == 800
    assert settings.RAG_CHUNK_OVERLAP == 100


def test_rag_exceptions():
    """Verify Phase 1 RAG exception hierarchy."""
    exc = SSRFProtectionException("Disallowed domain attempted")
    assert isinstance(exc, CrawlerException)
    assert isinstance(exc, RAGException)
    assert exc.message == "Disallowed domain attempted"

    emb_exc = EmbeddingException("Embedding API timeout")
    assert isinstance(emb_exc, RAGException)
    assert emb_exc.message == "Embedding API timeout"


def test_rag_models_serialization():
    """Verify Phase 1 Document, Chunk, CrawlRun, SourceCitation model serialization."""
    doc = RAGDocumentModel(
        url="https://webandcrafts.com/services",
        canonical_url="https://webandcrafts.com/services",
        title="WAC Services",
        description="Services provided by Web and Crafts",
        content_hash="sha256hash12345",
        domain="webandcrafts.com",
        word_count=450,
    )
    doc_dict = doc.model_dump()
    assert doc_dict["domain"] == "webandcrafts.com"
    assert doc_dict["status"] == "active"
    assert doc_dict["word_count"] == 450

    chunk = RAGChunkModel(
        document_id="doc123",
        chunk_index=0,
        content="WAC offers AI software engineering services.",
        title="WAC Services",
        heading_path=["Services", "AI Engineering"],
        url="https://webandcrafts.com/services",
        canonical_url="https://webandcrafts.com/services",
        embedding=[0.1] * 768,
        content_hash="sha256hash12345",
    )
    chunk_dict = chunk.model_dump()
    assert chunk_dict["document_id"] == "doc123"
    assert len(chunk_dict["embedding"]) == 768
    assert chunk_dict["heading_path"] == ["Services", "AI Engineering"]

    crawl_run = CrawlRunModel(
        urls_discovered=10,
        urls_crawled=8,
        documents_changed=2,
    )
    assert crawl_run.status == "running"
    assert crawl_run.urls_discovered == 10

    citation = SourceCitation(
        title="WAC Services",
        url="https://webandcrafts.com/services",
        heading="AI Engineering",
        score=0.92,
    )
    assert citation.score == 0.92

    rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        context="WAC provides AI engineering",
        sources=[citation],
        retrieval_score=0.92,
    )
    assert len(rag_result.sources) == 1
    assert rag_result.sources[0].title == "WAC Services"


@pytest.mark.anyio
async def test_rag_repository_instantiation():
    """Verify repository instantiation."""
    from app.repositories.rag_repository import (
        RAGDocumentRepository,
        RAGChunkRepository,
        CrawlRunRepository,
    )
    # Testing repository instantiation without requiring active Mongo connection
    doc_repo = RAGDocumentRepository.__new__(RAGDocumentRepository)
    chunk_repo = RAGChunkRepository.__new__(RAGChunkRepository)
    crawl_repo = CrawlRunRepository.__new__(CrawlRunRepository)
    assert doc_repo is not None
    assert chunk_repo is not None
    assert crawl_repo is not None
