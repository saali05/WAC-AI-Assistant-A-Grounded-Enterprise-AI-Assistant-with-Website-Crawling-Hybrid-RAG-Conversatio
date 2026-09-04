# WAC AI Assistant --- System Architecture

## Overview

WAC AI Assistant is an enterprise conversational AI application built
around FastAPI, React, MongoDB, Gemini/Groq providers, website crawling,
hybrid RAG, function calling, voice interaction, conversation
management, and usage analytics.

The assistant is designed to answer Web and Craft (WAC)-related
questions and ground company-specific answers in the WAC knowledge base.

## High-Level Flow

``` text
User
  |
  v
React Frontend
  |
  v
FastAPI
  |
  v
AIService
  |
  +--------------------+
  |                    |
Text Chat           Voice Chat
  |                    |
Gemini / Groq       Gemini Live
  |
Function Calling when knowledge is required
  |
  v
ToolExecutor
  |
  v
search_wac_knowledge
  |
  v
RAGService
  |
  +-------------------+
  |                   |
Vector Search     Keyword Search
  |                   |
  +---------+---------+
            |
        Reranking
            |
        Context Builder
            |
            v
       Grounded Context
            |
            v
       Gemini / model
            |
            v
       Final Response
```

## Main Components

### Frontend

Provides text chat, model selection, voice interaction, sources, session
experience, and analytics UI.

### FastAPI API Layer

Provides HTTP endpoints and connects the frontend with application
services.

### AI Layer

Contains provider abstraction, Gemini and Groq providers, AI schemas,
usage tracking, and function/tool handling.

### RAG Layer

Contains website crawling, HTML extraction, metadata extraction,
semantic chunking, embeddings, indexing, vector search, keyword search,
hybrid retrieval, reranking, relevance validation, and context
construction.

### Data Layer

MongoDB stores conversations, messages, memory, RAG documents, RAG
chunks, crawl runs, and usage information.

## Provider Architecture

``` text
AIService
   |
ProviderFactory
   |
+--+------+
|         |
Gemini   Groq
```

Normal text chat can use Gemini or Groq. Gemini Live is a separate
real-time voice path controlled by `GEMINI_LIVE_MODEL`.

## Knowledge Ingestion

``` text
WAC Website
    |
WebCrawler
    |
HTML / Metadata Extraction
    |
Semantic Chunking
    |
Gemini Embedding
    |
MongoDB
```

The indexer checks content hashes so unchanged documents can be skipped
and changed documents can be versioned and re-indexed.

## Query Architecture

``` text
User Question
    |
    v
AIService
    |
    v
LLM
    |
    +---- no tool needed ----> Response
    |
    +---- knowledge needed --> Function Call
                                  |
                                  v
                             ToolExecutor
                                  |
                                  v
                              RAGService
                                  |
                                  v
                           Retrieved Context
                                  |
                                  v
                                LLM
                                  |
                                  v
                            Final Answer
```

## Design Principles

Responsibilities are separated between API routers, services, AI
providers, tools, RAG components, and repositories. This makes the
application easier to test, extend, and maintain.
