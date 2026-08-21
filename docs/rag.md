# Grounded RAG Subsystem via Website Crawling

## Overview

The Grounded RAG System turns the WAC AI Assistant into an enterprise-grade assistant grounded directly in official website content (`webandcrafts.com`).

---

## Architecture

```
User Query
   │
   ▼
Chat API / ChatService
   │
   ▼
WAC Relevance Gate (Out-of-Domain Refusal check)
   │
   ▼
Query Rewriter (Session memory context)
   │
   ▼
Hybrid Search
   ├── Vector Search (MongoDB Atlas $vectorSearch / Cosine fallback)
   └── Keyword Search (Full-Text Search / Regex)
   │
   ▼
Reciprocal Rank Fusion (RRF) & Fusion Reranker
   │
   ▼
Relevance Threshold Check (RAG_MIN_RELEVANCE_SCORE)
   │
   ▼
Context Builder (Formatted WAC Knowledge Context)
   │
   ▼
PromptBuilder (SYSTEM, RESPONSE, COMPANY, MEMORY, RAG KNOWLEDGE)
   │
   ▼
AIService (Gemini / Groq)
   │
   ▼
Grounded Answer + Source Citations Card Display
```

---

## Crawler Architecture & Security

1. **URL Normalization (`url_normalizer.py`):**
   - Canonicalizes scheme, host, path, and removes tracking params (`utm_*`, `fbclid`, `gclid`).
   - **SSRF Protection:** Rejects `localhost`, `127.0.0.1`, private IP ranges (`10.x`, `172.16-31.x`, `192.168.x`, `169.254.x`), metadata endpoints, and invalid URI schemes (`file://`, `javascript:`).
   - Domain Allowlist check (`RAG_ALLOWED_DOMAINS`).

2. **Robots.txt (`robots.py`):**
   - Parses `/robots.txt` per domain, respects `Crawl-delay` and extracts sitemaps.

3. **Sitemap Parsing (`sitemap.py`):**
   - Parses `<urlset>` and `<sitemapindex>` XML structures and sorts URLs by `<lastmod>` timestamps.

4. **Heading-Aware Semantic Chunker (`semantic_chunker.py`):**
   - Splits content into 400–800 token chunks with 50–100 token overlap.
   - Retains structural heading hierarchy (`heading_path`).

5. **Document Versioning & Indexing (`indexer.py`):**
   - SHA-256 content hashing (`old_hash == new_hash` -> skips re-embedding).
   - Increments document version and updates active chunks in `rag_chunks`.

---

## MongoDB Atlas Vector Search Index Setup

Create an Atlas Vector Search index on collection `rag_chunks`:

Index Name: `vector_index`

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "status",
      "type": "filter"
    }
  ]
}
```

---

## Environment Configuration

```ini
RAG_ENABLED=True
RAG_ALLOWED_DOMAINS=webandcrafts.com,www.webandcrafts.com
RAG_EMBEDDING_MODEL=text-embedding-004
RAG_EMBEDDING_DIMENSIONS=768
RAG_VECTOR_WEIGHT=0.7
RAG_KEYWORD_WEIGHT=0.3
RAG_MIN_RELEVANCE_SCORE=0.65
RAG_TOP_K_VECTOR=20
RAG_TOP_K_KEYWORD=20
RAG_TOP_K_FINAL=5
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_MAX_PAGES=1000
RAG_MAX_DEPTH=5
RAG_REQUEST_TIMEOUT=15
RAG_CRAWL_DELAY=1.0
RAG_CONCURRENCY=5
```

---

## Admin Endpoints

- `POST /rag/crawl`: Trigger manual website crawl run.
- `GET /rag/crawl/status`: Check current or latest crawl status and stats.
- `POST /rag/reindex`: Re-chunk and re-embed active documents.
- `GET /rag/documents`: List indexed documents.
- `GET /rag/documents/{id}`: View document details.
- `DELETE /rag/documents/{id}`: Delete indexed document & chunks.
