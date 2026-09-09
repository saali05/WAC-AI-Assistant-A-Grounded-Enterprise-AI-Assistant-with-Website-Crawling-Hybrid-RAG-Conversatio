# WAC AI Assistant — Setup & Deployment Guide

This guide provides step-by-step instructions for configuring, running, testing, and deploying the WAC AI Assistant locally and in production.

---

## 1. Prerequisites

Ensure your development environment meets the following specifications:
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.11 or higher
- **Node.js**: Version 18.0.0 or higher with `npm`
- **MongoDB**: Version 6.0+ (Local Community Edition or MongoDB Atlas with Vector Search index support)
- **API Keys**:
  - Google Gemini API Key (Required for Gemini text generation and Gemini Live voice)
  - Groq API Key (Required for Groq Llama 3 generation)

---

## 2. Backend Setup

### A. Clone and Virtual Environment

```bash
# Navigate to project directory
cd "AI Chat_Bot"

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### B. Dependency Installation

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 3. Environment Configuration (`.env`)

Create a `.env` file in the project root directory:

```ini
# ============================================================
# APPLICATION SETTINGS
# ============================================================
APP_NAME="WAC AI Assistant"
APP_VERSION="1.0.0"
DEBUG=False

# ============================================================
# AI GENERATION PROVIDERS
# ============================================================
DEFAULT_PROVIDER="gemini"

# Gemini Generation
GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_LIVE_MODEL="gemini-3.1-flash-live-preview"

# Groq Generation
GROQ_API_KEY="gsk_YourGroqApiKeyHere"
GROQ_MODEL="llama-3.3-70b-versatile"

# ============================================================
# EMBEDDING PROVIDER CONFIGURATION
# ============================================================
# Options: "gemini" (Hosted API) | "local" (Local BGE-base via sentence-transformers)
RAG_EMBEDDING_PROVIDER="gemini"
LOCAL_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
LOCAL_EMBEDDING_DIMENSIONS=768

# ============================================================
# PRICING & TELEMETRY
# ============================================================
AI_PRICING_MODE="paid"
AI_CURRENCY="USD"
USE_LANGCHAIN_PIPELINE=False

# Pricing per 1M tokens (USD)
GEMINI_INPUT_PRICE_PER_1M=1.50
GEMINI_OUTPUT_PRICE_PER_1M=7.50
GROQ_INPUT_PRICE_PER_1M=0.15
GROQ_OUTPUT_PRICE_PER_1M=0.60
GEMINI_LIVE_TEXT_INPUT_PRICE_PER_1M=0.75
GEMINI_LIVE_TEXT_OUTPUT_PRICE_PER_1M=4.50
GEMINI_LIVE_AUDIO_INPUT_PRICE_PER_1M=3.00
GEMINI_LIVE_AUDIO_OUTPUT_PRICE_PER_1M=12.00
GEMINI_AUDIO_TOKENS_PER_SECOND=25

# ============================================================
# DATABASE SETTINGS
# ============================================================
MONGODB_URI="mongodb://localhost:27017"
DATABASE_NAME="wac_ai"

# ============================================================
# RAG & CRAWLING CONFIGURATION
# ============================================================
ALLOWED_DOMAINS="webandcrafts.com"
START_URLS="https://webandcrafts.com"
MAX_PAGES_TO_CRAWL=50
CRAWL_CONCURRENCY=5
CRAWL_DELAY=1.0

# Document Upload Limits
UPLOAD_DIR="./uploads"
MAX_FILE_SIZE=10485760
```

---

## 4. MongoDB Database & Index Setup

Connect to your MongoDB instance and ensure the database `wac_ai` is configured with the necessary search indexes:

### 1. MongoDB Atlas Vector Search Index (for `rag_chunks` collection)
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

### 2. MongoDB Full-Text Index (for `rag_chunks` collection)
```javascript
use wac_ai;
db.rag_chunks.createIndex({
  content: "text",
  title: "text",
  heading_path: "text"
}, {
  weights: {
    title: 10,
    heading_path: 5,
    content: 1
  },
  name: "rag_chunks_text_search"
});
```

---

## 5. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install
```

---

## 6. Running the Application

### A. Start the Backend Server

```bash
# From workspace root with venv activated:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### B. Start the Frontend Development Server

```bash
# From frontend/ directory:
npm run dev
```
Access the client UI at: `http://localhost:5173`.

---

## 7. Verification & Automated Testing

### Run Pytest Suite
```bash
python -m pytest -q
```
Expected output: **161 passed** (0 failures).

### Run Compilation Bytecode Check
```bash
python -m compileall app
```
Expected output: **0 compilation errors across all subpackages**.

### Trigger an Ingestion Crawl
```bash
curl -X POST http://127.0.0.1:8000/rag/crawl
```

---

## 8. Production Build

To prepare the frontend for production hosting:

```bash
cd frontend
npm run build
```

This validates TypeScript types (`tsc -b`) and bundles static assets into `frontend/dist/`. The static assets can be served via Nginx, Caddy, or mounted directly in FastAPI.
