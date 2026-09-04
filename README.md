# WAC AI Assistant / Chatbot with Grounded RAG via Website Crawling

AI Assistant for Web and Crafts (WAC) powered by FastAPI, MongoDB, Gemini, Groq, Gemini Live Voice, and a Grounded Website Crawling RAG pipeline.

---

## Features

- **FastAPI Backend:** Async API with Chat, Voice, Analytics, and RAG Admin routers.
- **React + TypeScript Frontend:** Glassmorphic UI with dynamic model selection and RAG source citation cards.
- **Website Crawling RAG Pipeline:**
  - Automated page discovery (Robots.txt & Sitemap parser).
  - Normalizer & SSRF Security protection.
  - HTML extraction & boilerplate cleaner.
  - Heading-aware semantic chunking.
  - Gemini Vector Embeddings (`text-embedding-001`).
  - Hybrid Search (Vector + Keyword Search with Reciprocal Rank Fusion).
  - Fusion Reranker & Min Relevance Thresholding.
  - WAC Domain Relevance Gate & Grounding Validator.
- **Provider Abstraction:** Gemini (`gemini-3.6-flash`), Groq (`openai/gpt-oss-120b`), Gemini Live (`gemini-3.1-flash-live-preview`).
- **Session Memory & Usage Analytics:** Conversation management, token tracking, and pricing analytics.

---

## Quick Start

### 1. Environment Setup

Copy `.env` and verify MongoDB connection details (`MONGODB_URI=mongodb://localhost:27017`).

### 2. Run Backend

```bash
.\venv\Scripts\python -m uvicorn app.main:app --reload
```

### 3. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Run Unit & RAG Test Suite

```bash
.\venv\Scripts\python -m pytest
```

### 5. Trigger Initial Crawl

```bash
curl -X POST http://localhost:8000/rag/crawl
```
