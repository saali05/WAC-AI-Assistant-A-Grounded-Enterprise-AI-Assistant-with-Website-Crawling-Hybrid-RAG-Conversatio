# WAC AI Assistant — Enterprise Grounded Conversational AI

An enterprise-grade, grounded conversational AI assistant designed for **Web and Crafts (WAC)**. Built with **FastAPI**, **React (Vite + TypeScript)**, **MongoDB Atlas (Hybrid Vector + Keyword Search)**, **Google Gemini 2.5 Flash / Groq Llama 3.3 70B Versatile**, and **Gemini Live Multimodal Voice**.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Detailed RAG Pipeline](#detailed-rag-pipeline)
  - [1. Domain Relevance Gate](#1-domain-relevance-gate)
  - [2. Website Crawler & Ingestion](#2-website-crawler--ingestion)
  - [3. Semantic Chunking & Indexing](#3-semantic-chunking--indexing)
  - [4. Query Rewriting & Intent Detection](#4-query-rewriting--intent-detection)
  - [5. Hybrid Vector + Keyword Retrieval](#5-hybrid-vector--keyword-retrieval)
  - [6. Reciprocal Rank Fusion (RRF)](#6-reciprocal-rank-fusion-rrf)
  - [7. Intent-Aware Fusion Reranking](#7-intent-aware-fusion-reranking)
  - [8. Company-Specific Evidence Sufficiency](#8-company-specific-evidence-sufficiency)
  - [9. Grounded Context Generation](#9-grounded-context-generation)
- [Gemini Live Voice Subsystem](#gemini-live-voice-subsystem)
- [Session Memory & Conversation Scope](#session-memory--conversation-scope)
- [Analytics & Cost Telemetry](#analytics--cost-telemetry)
- [Embedding & Generation Provider Independence](#embedding--generation-provider-independence)
- [Installation & Quick Start](#installation--quick-start)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Variables Reference](#environment-variables-reference)
  - [MongoDB Atlas Index Setup](#mongodb-atlas-index-setup)
- [Running the Application](#running-the-application)
- [Running Test Suite](#running-test-suite)
- [Production Build](#production-build)
- [Embedding Migration & Vector Space Isolation](#embedding-migration--vector-space-isolation)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

The **WAC AI Assistant** is a production-ready conversational platform engineered to represent Web and Crafts with absolute grounding, zero out-of-domain hallucinations, real-time voice streaming, and fine-grained session analytics.

The system indexes WAC's digital ecosystem directly through automated website crawling, processes documents into semantic chunks, and evaluates all user interactions through a mandatory RAG pipeline with strict company-evidence gating.

---

## Problem Statement

Traditional enterprise chatbots often suffer from three critical failure modes:
1. **Hallucination & Scope Creep**: Answering general world-knowledge questions (e.g. "What is the capital of France?") or claiming capabilities/technologies the company does not actually offer.
2. **False Technology Attribution**: Interpreting generic educational blog posts (e.g., "Node.js vs PHP: A Comparison") as proof that the company uses or endorses a technology for its client services.
3. **Provider Lock-in & Quota Fragility**: Tight coupling between embedding models and text generation engines, causing complete system outages if an API quota is exhausted.

**WAC AI Assistant addresses all three issues** through a multi-stage validation architecture, decoupled provider layers, and intent-aware evidence sufficiency gates.

---

## Key Features

- **Mandatory Grounded RAG Pipeline**: Ensures no LLM generation occurs without verified, retrieved context from WAC's official knowledge base.
- **WAC Relevance Gate**: Automatically filters non-WAC questions at the boundary with zero embedding or retrieval cost.
- **Hybrid Vector + Keyword Search**: Atlas Vector Search combined with MongoDB full-text search fused via Reciprocal Rank Fusion ($k=60$).
- **Intent-Aware Reranker with Company Grounding**: Prioritizes service and case-study evidence over generic articles and strictly validates company relationship markers.
- **Decoupled Provider Matrix**:
  - **Generation**: Google Gemini 2.5 Flash (`gemini`) or Groq Llama 3.3 70B Versatile (`groq`).
  - **Embeddings**: Google Gemini (`gemini-embedding-001`) or Local BGE (`BAAI/bge-base-en-v1.5` via `sentence-transformers`).
- **Gemini Live Multimodal Voice**: Real-time bidirectional Web Audio streaming over WebSockets with server-side tool calling via LangChain `WACRetriever`.
- **Session-Scoped Usage & Cost Telemetry**: Exact input/output token counts, audio turn durations, latency metrics, and USD pricing breakdown.
- **Glassmorphic Responsive UI**: Built with React 19, TypeScript, Vite, Tailwind CSS, and Lucide icons.

---

## System Architecture

```
User Browser (React 19 / TypeScript)
  ├── Text Chat UI (Model Selector: Gemini / Groq)
  ├── Gemini Live Voice UI (Microphone PCM 16kHz Stream)
  └── Session Analytics Sidebar
       │
       ▼ HTTP / WebSocket
FastAPI Application Layer (`app/main.py`)
  ├── `/chat` (ChatService)
  ├── `/voice/token`, `/voice/tool`, `/voice/message` (VoiceRouter)
  ├── `/conversations/{id}/analytics` (UsageService)
  └── `/rag/crawl`, `/rag/reindex` (CrawlService / Indexer)
       │
       ▼
AIService Layer (`app/ai/service.py`)
  ├── WACRelevanceGate (Domain validation)
  ├── ProviderFactory (GeminiProvider / GroqProvider)
  └── UsageService (Token, latency & cost telemetry)
       │
       ▼
Mandatory RAG Engine (`app/services/rag_service.py`)
  ├── QueryRewriter (Contextual history & intent detection)
  ├── HybridSearch (Atlas Vector + MongoDB Keyword Search)
  ├── Reciprocal Rank Fusion (RRF)
  ├── FusionReranker (Domain & evidence prioritization)
  └── Evidence Sufficiency Validator (_has_company_relationship_evidence)
       │
       ▼
Storage Layer (MongoDB)
  ├── `conversations`, `messages`, `memories`
  ├── `rag_documents`, `rag_chunks` (Vector + Text Indexes)
  └── `chat_usage`, `crawl_runs`
```

---

## Detailed RAG Pipeline

```
User Query
    │
    ▼
[ WAC Relevance Gate ] ──(Non-WAC)──► Standard Refusal (0 API Calls)
    │ (WAC Query)
    ▼
[ Query Rewriter ] ──► Expanded Query + Intent (SERVICES, COMPANY_TECH, etc.)
    │
    ├──► [ Vector Search ] (Atlas Vector Search / 768d Cosine)
    └──► [ Keyword Search ] (MongoDB Full-Text Search)
    │
    ▼
[ Reciprocal Rank Fusion (RRF) ] ──► Score = Σ 1 / (60 + rank)
    │
    ▼
[ Fusion Reranker ] ──► Content Scoring + Service Boost + URL Diversity
    │
    ▼
[ Evidence Sufficiency Gate ] ──(Insufficient)──► Grounded Refusal
    │ (Sufficient Evidence)
    ▼
[ Grounded LLM Generation ] (Gemini 2.5 Flash / Groq Llama 3.3 70B)
    │
    ▼
Answer + Verifiable Source Cards (Title, URL, Heading, Score)
```

### 1. Domain Relevance Gate
[`WACRelevanceGate`](file:///c:/WAC/AI%20Chat_Bot/app/rag/validation/relevance.py) inspects incoming prompts using keyword, domain, and topical pattern heuristics. Queries completely unrelated to WAC (e.g., general trivia, foreign companies) are rejected immediately with a friendly refusal message, consuming zero embedding or LLM tokens.

### 2. Website Crawler & Ingestion
[`WebCrawler`](file:///c:/WAC/AI%20Chat_Bot/app/rag/crawler/crawler.py) respects `robots.txt`, processes sitemaps, verifies SSRF safety, canonicalizes URLs, and strips boilerplate HTML headers, footers, and scripts.

### 3. Semantic Chunking & Indexing
[`SemanticChunker`](file:///c:/WAC/AI%20Chat_Bot/app/rag/chunking/semantic_chunker.py) splits extracted text while maintaining header hierarchies (`heading_path`) and contextual overlap. Documents are versioned in MongoDB; unchanged content hashes are skipped during incremental crawls.

### 4. Query Rewriting & Intent Detection
[`QueryRewriter`](file:///c:/WAC/AI%20Chat_Bot/app/rag/retrieval/query_rewriter.py) performs conversational coreference resolution (e.g., resolving *"What technologies do they use for that?"* after asking about ecommerce) and categorizes query intent (`COMPANY_TECH`, `SERVICES`, `CAREERS`, `COMPANY_INFO`, `GENERAL`).

### 5. Hybrid Vector + Keyword Retrieval
[`HybridSearch`](file:///c:/WAC/AI%20Chat_Bot/app/rag/retrieval/hybrid_search.py) queries both:
- **Vector Search**: Computes cosine similarity against MongoDB Atlas vector indexes.
- **Keyword Search**: Performs MongoDB full-text search against chunk content, headings, and titles.

### 6. Reciprocal Rank Fusion (RRF)
Vector and lexical candidate lists are fused using standard RRF:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$

### 7. Intent-Aware Fusion Reranking
[`FusionReranker`](file:///c:/WAC/AI%20Chat_Bot/app/rag/retrieval/reranker.py) applies contextual adjustments:
- Promotes official service pages (`/services/*`) and case studies.
- Applies strict 1-chunk-per-URL diversity in top rank positions.
- Demotes ungrounded third-party comparison articles for company queries.

### 8. Company-Specific Evidence Sufficiency
[`RAGService._evaluate_evidence_sufficiency`](file:///c:/WAC/AI%20Chat_Bot/app/services/rag_service.py) enforces that generic educational blogs (e.g., discussing "Node.js vs PHP") cannot be used as proof that WAC uses a technology unless accompanied by explicit company relationship markers (`"our developers"`, `"we provide"`, `"WAC builds"`, or official service URLs).

### 9. Grounded Context Generation
Retrieved chunks above the minimum confidence threshold (`0.50`) are packaged with strict system constraints. The LLM generates the response accompanied by source metadata.

---

## Gemini Live Voice Subsystem

WAC AI includes a real-time bidirectional voice interface powered by Google Gemini Live:
1. **Authentication (`GET /voice/token`)**: Issues an ephemeral OAuth token constrained to `GEMINI_LIVE_MODEL` and registering the `search_wac_knowledge` tool.
2. **WebSocket Audio Streaming**: Frontend captures 16kHz PCM audio and streams real-time buffers to Gemini Live.
3. **Server-Side Function Calling (`POST /voice/tool`)**: When factual company information is needed, Gemini Live calls `search_wac_knowledge`, which executes [`WACRetriever`](file:///c:/WAC/AI%20Chat_Bot/app/langchain/retrievers/wac_retriever.py) on the backend.
4. **Usage Persistence (`POST /voice/message`)**: On turn completion, audio duration, turn tokens, latency, and session IDs are persisted to MongoDB via `UsageService`.

---

## Embedding & Generation Provider Independence

Embedding generation and text generation are completely decoupled:

| Embedding Provider (`RAG_EMBEDDING_PROVIDER`) | Model | Dimensions | Hosting |
| :--- | :--- | :--- | :--- |
| **`gemini`** (Active Production Default) | `gemini-embedding-001` | 768 | Google GenAI API |
| **`local`** (Local Zero-Quota Fallback) | `BAAI/bge-base-en-v1.5` | 768 | Local CPU/GPU (`sentence-transformers`) |

| Generation Provider (`DEFAULT_PROVIDER` / Dropdown) | Model | Purpose |
| :--- | :--- | :--- |
| **`gemini`** | `gemini-2.5-flash` | Ultra-fast multimodal chat |
| **`groq`** | `llama-3.3-70b-versatile` | Open-weights low-latency inference |

---

## Installation & Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+ & npm**
- **MongoDB 6.0+** (Local or MongoDB Atlas with Vector Search support)
- **Google Gemini API Key** and/or **Groq API Key**

### Backend Setup

```bash
# 1. Clone repository
git clone https://github.com/saali05/AI-Chat_Bot.git
cd "AI Chat_Bot"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Environment Variables Reference

Edit `.env` in the repository root:

```ini
# Application
APP_NAME="WAC AI Assistant"
APP_VERSION="1.0.0"
DEBUG=False

# Generation Providers
DEFAULT_PROVIDER="gemini"
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_LIVE_MODEL="gemini-3.1-flash-live-preview"

GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="llama-3.3-70b-versatile"

# Embedding Provider Configuration
RAG_EMBEDDING_PROVIDER="gemini"
LOCAL_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
LOCAL_EMBEDDING_DIMENSIONS=768

# Pricing Mode
AI_PRICING_MODE="paid"
AI_CURRENCY="USD"

# MongoDB
MONGODB_URI="mongodb://localhost:27017"
DATABASE_NAME="wac_ai"

# RAG & Crawling
ALLOWED_DOMAINS="webandcrafts.com"
START_URLS="https://webandcrafts.com"
MAX_PAGES_TO_CRAWL=50
```

### MongoDB Atlas Index Setup

Ensure the following indexes exist in your MongoDB `wac_ai` database:

1. **Vector Index on `rag_chunks`** (Atlas Search):
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 768,
         "similarity": "cosine"
       }
     ]
   }
   ```
2. **Text Index on `rag_chunks`**:
   ```javascript
   db.rag_chunks.createIndex({
     content: "text",
     title: "text",
     "heading_path": "text"
   })
   ```

---

## Running the Application

### Start Backend
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend
```bash
cd frontend
npm run dev
```

The application will be accessible at `http://localhost:5173`.

---

## Running Test Suite

```bash
# Run full pytest suite (161 tests)
python -m pytest -q

# Run Python compilation check
python -m compileall app
```

---

## Production Build

```bash
cd frontend
npm run build
```
Build artifacts will be compiled to `frontend/dist/`.

---

## Embedding Migration & Vector Space Isolation

> [!WARNING]
> **Semantic Vector Space Incompatibility:**
> Even though both `gemini-embedding-001` and `BAAI/bge-base-en-v1.5` use 768 dimensions, their latent geometric vector spaces are **completely incompatible**. You cannot query a Gemini-embedded chunk corpus with a Local BGE query embedding.

### Current Production Status:
* **Active Production Provider**: `gemini` (`gemini-embedding-001`)
* **Migration Status**: **NOT RUN / PAUSED** (The database is populated with Gemini embeddings).

If you wish to migrate the database to Local BGE embeddings in the future:
```bash
# 1. Perform a dry-run test
python scripts/migrate_embeddings.py --provider local --dry-run

# 2. Execute migration with explicit confirmation
python scripts/migrate_embeddings.py --provider local --confirm
```
For detailed migration procedures and rollback strategies, see [docs/embedding-migration.md](file:///c:/WAC/AI%20Chat_Bot/docs/embedding-migration.md).

---

## Known Limitations

1. **Free-Tier Gemini Embedding Quota**: Rapid consecutive indexing of hundreds of pages under Google GenAI Free Tier may encounter rate limits (`429 Quota Exceeded`). Local BGE provides a zero-quota alternative.
2. **Real-Time Live WebSockets**: Gemini Live WebSocket token usage headers are dependent on Google's WebSocket metadata availability; when unavailable, analytics safely report `"Unavailable"` without fabricating estimates.

---

## License

Proprietary — Developed for Web and Crafts (WAC). All rights reserved.
