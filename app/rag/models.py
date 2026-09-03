from datetime import datetime, UTC
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RAGDocumentModel(BaseModel):
    """Document model representing a crawled webpage in rag_documents collection."""
    id: Optional[str] = Field(None, alias="_id")
    url: str
    canonical_url: str
    title: str = ""
    description: str = ""
    content_hash: str
    version: int = 1
    status: Literal["active", "inactive", "archived"] = "active"
    source_type: str = "website"
    domain: str
    language: str = "en"
    word_count: int = 0
    first_crawled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_crawled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    http_status: int = 200
    error_message: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class RAGChunkModel(BaseModel):
    """Chunk model representing a text chunk with vector embedding in rag_chunks collection."""
    id: Optional[str] = Field(None, alias="_id")
    document_id: str
    chunk_index: int
    content: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    url: str
    canonical_url: str
    embedding: list[float] = Field(default_factory=list)
    embedding_model: str = "text-embedding-004"
    embedding_dimensions: int = 768
    content_hash: str
    version: int = 1
    status: Literal["active", "inactive"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(populate_by_name=True)


class CrawlRunError(BaseModel):
    url: str
    error: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CrawlRunModel(BaseModel):
    """Crawl run metadata model in crawl_runs collection."""
    id: Optional[str] = Field(None, alias="_id")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: Optional[datetime] = None
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    urls_discovered: int = 0
    urls_crawled: int = 0
    documents_changed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    errors: list[CrawlRunError] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class SourceCitation(BaseModel):
    """Structured citation metadata returned to API & frontend consumers."""
    title: str
    url: str
    heading: Optional[str] = None
    score: float = 0.0
    canonical_url: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Internal retrieved chunk with scoring metadata."""
    chunk_id: str
    document_id: str
    content: str
    title: str
    heading_path: list[str]
    url: str
    canonical_url: str
    score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    fusion_score: Optional[float] = None
    reranked_score: Optional[float] = None
    retrieval_confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RAGResult(BaseModel):
    """Complete RAG retrieval & context generation result."""
    is_relevant: bool = True
    has_context: bool = False
    context: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)
    retrieval_score: float = 0.0
    refusal_reason: Optional[str] = None
