from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    start_urls: Optional[list[str]] = Field(default=None, description="Optional list of seed URLs to crawl.")


class CrawlStatusResponse(BaseModel):
    run_id: Optional[str] = None
    status: str = "idle"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    urls_discovered: int = 0
    urls_crawled: int = 0
    documents_changed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    errors: list[dict] = Field(default_factory=list)


class DocumentListItem(BaseModel):
    id: str
    url: str
    canonical_url: str
    title: str = ""
    domain: str = ""
    status: str = "active"
    word_count: int = 0
    last_crawled_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem] = Field(default_factory=list)
    total: int = 0
