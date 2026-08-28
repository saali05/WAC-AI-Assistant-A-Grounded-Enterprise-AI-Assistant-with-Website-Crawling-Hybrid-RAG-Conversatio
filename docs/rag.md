# WAC AI Assistant --- RAG

## Purpose

Retrieval-Augmented Generation (RAG) supplies grounded WAC knowledge to
the language model.

The system crawls the WAC website, extracts useful content, chunks it,
generates embeddings, stores searchable chunks in MongoDB, and retrieves
relevant information for user questions.

## Ingestion Pipeline

``` text
WAC Website
    |
    v
WebCrawler
    |
    v
HTML Extraction
    |
    v
Metadata Extraction
    |
    v
Semantic Chunker
    |
    v
Gemini Embedding Model
    |
    v
MongoDB rag_chunks
```

## Retrieval Pipeline

``` text
User Query
    |
    v
Query Rewriter
    |
    v
Hybrid Search
   /   /   Vector Keyword
Search Search
  \   /
   \ /
 Fusion / Reranking
      |
      v
Relevance Threshold
      |
      v
Context Builder
      |
      +---- Context
      |
      +---- Sources
```

## Website Crawling

`WebCrawler` discovers pages under configured allowed domains.
`CrawlService` tracks crawl runs, URLs discovered/crawled, documents
changed/skipped, chunks created, and errors.

## Document Indexing

For a new document:

``` text
Document -> Chunks -> Embeddings -> MongoDB
```

For an existing document, the content hash is checked.

If unchanged:

``` text
Skip re-embedding
Update crawl timestamp
```

If changed:

``` text
Create new version
Deactivate previous active chunks
Re-chunk
Generate embeddings
Store new chunks
Update document
```

## Embeddings

The project uses:

``` text
gemini-embedding-001
```

with the configured embedding dimension:

``` text
768
```

An embedding converts text into a numerical vector.

There are two uses of the same embedding process:

1.  During indexing, each stored chunk is embedded.
2.  During retrieval, the user query is embedded.

The query vector is compared with stored vectors to find semantically
similar chunks.

## Hybrid Search

### Vector Search

Finds semantically similar content using embeddings.

### Keyword Search

Finds content using textual matching. This is useful for exact names,
technologies, services, and other lexical terms.

### Hybrid Retrieval

Combines both signals to improve retrieval quality.

## Reranking

Retrieved candidates are reranked so the most useful chunks are placed
first before context is sent to the model.

## Relevance Threshold

The top retrieval score is compared with the configured minimum
relevance score. If the result is not sufficiently relevant, the system
returns a controlled knowledge-base refusal rather than presenting weak
retrieval as reliable information.

## Context and Sources

`ContextBuilder` converts selected chunks into model-ready context and
source information. The frontend can display the supporting sources with
the response.

## Function Calling + RAG

RAG is exposed through the tool:

``` text
search_wac_knowledge
```

Function calling does not replace RAG.

``` text
LLM
 |
 | function call
 v
ToolExecutor
 |
 v
search_wac_knowledge
 |
 v
RAGService
 |
 v
Hybrid Retrieval
 |
 v
Grounded Context
 |
 v
LLM
 |
 v
Final Answer
```

## Reindexing

Reindexing is useful when chunking logic, embedding configuration, or
indexed knowledge needs to be regenerated. Large embedding operations
must account for provider request quotas.

## Benefits

RAG provides current website knowledge, grounding, source-backed
answers, easier knowledge updates, and a controlled company-knowledge
boundary.
