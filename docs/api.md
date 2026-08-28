# WAC AI Assistant --- API Documentation

## Overview

The backend uses FastAPI to expose endpoints for chat, voice, RAG
operations, and related application functionality.

The exact endpoint set should remain synchronized with the current
FastAPI routers.

## Chat

### POST `/chat`

Sends a normal text-chat request.

``` text
Frontend
   |
POST /chat
   |
Chat API
   |
ChatService
   |
AIService
   |
Gemini / Groq
   |
Function calling when required
   |
Final response
```

## RAG Crawl Status

### GET `/rag/crawl/status`

Returns the latest or requested crawl-run status.

Typical information includes:

-   Run ID
-   Status
-   Start time
-   Finish time
-   URLs discovered
-   URLs crawled
-   Documents changed
-   Documents skipped
-   Chunks created
-   Errors

## RAG Documents

### GET `/rag/documents`

Provides information about indexed RAG documents, such as:

-   Document ID
-   URL
-   Canonical URL
-   Title
-   Domain
-   Status
-   Word count
-   Crawl timestamp

This is useful for validating website crawling and indexing.

## RAG Reindex

### POST `/rag/reindex`

Regenerates/reindexes active RAG knowledge according to the current
implementation.

Embedding provider quotas can cause this endpoint to return a rate-limit
error such as:

``` text
429 RESOURCE_EXHAUSTED
```

A provider quota error is different from a Python or RAG logic error.

## Voice Token

### GET `/voice/token`

Creates a short-lived token for a Gemini Live session.

The endpoint uses:

``` text
GEMINI_API_KEY
GEMINI_LIVE_MODEL
```

The Live configuration includes audio settings and the WAC-specific
system instruction.

## Save Voice Message

### POST `/voice/message`

Persists a voice conversation turn and records usage.

The request can include:

-   Conversation ID
-   User message
-   Assistant message
-   Input audio duration
-   Output audio duration
-   Input tokens
-   Output tokens
-   Latency
-   Live session ID

## API Validation

FastAPI and Pydantic validate API request structures before business
logic processes them.

## Local Testing

Compile the application:

``` powershell
python -m compileall app
```

Open Swagger:

``` powershell
curl.exe http://localhost:8000/docs
```

Check crawl status:

``` powershell
curl.exe http://localhost:8000/rag/crawl/status
```

Inspect documents:

``` powershell
curl.exe "http://localhost:8000/rag/documents?limit=5"
```

Trigger reindex:

``` powershell
curl.exe -X POST "http://localhost:8000/rag/reindex"
```

For complete validation, also verify MongoDB state and the frontend
behavior.

## API Architecture

``` text
HTTP Request
    |
FastAPI Router
    |
Service Layer
    |
Repository / AI Provider / RAG
    |
MongoDB or External AI API
    |
Response
```

API routing is separated from business logic so services and
repositories can be tested independently.
