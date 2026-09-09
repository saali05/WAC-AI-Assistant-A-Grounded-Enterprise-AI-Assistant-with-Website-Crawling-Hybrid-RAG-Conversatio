# WAC AI Assistant — Retrieval-Augmented Generation (RAG) Architecture

## 1. Overview & Purpose

The Retrieval-Augmented Generation (RAG) pipeline is the core grounding mechanism of the WAC AI Assistant. Its purpose is to deliver accurate, up-to-date, verifiable factual knowledge about Web and Crafts (WAC) directly to the language model during inference.

The system prevents hallucinations by enforcing a **Mandatory Grounded RAG** contract:
- The language model never answers company questions using unverified pre-trained weights.
- Generic blog articles cannot satisfy technology usage claims without explicit company relationship markers.
- Every verified claim is accompanied by precise, traceable source citations.

---

## 2. Ingestion & Crawling Pipeline

```
WAC Web Ecosystem (https://webandcrafts.com)
     │
     ▼
[ WebCrawler ] ──► Respects robots.txt, Sitemaps, SSRF Filtering, Allowed Domains
     │
     ▼
[ HTML & Metadata Extraction ] ──► Strips boilerplate, extracts title & headings
     │
     ▼
[ Semantic Chunker ] ──► Heading-aware hierarchical chunking + overlap
     │
     ▼
[ Content Hash Comparison ]
     ├── Unchanged: Skip re-embedding, update crawl timestamp
     └── Changed / New: Version document, generate embeddings
          │
          ▼
[ Embedding Provider ] (Gemini text-embedding-001 or Local BGE 768d)
          │
          ▼
MongoDB Storage (`rag_documents`, `rag_chunks`)
```

### Crawl Service & Security
- **Domain Whitelisting**: Restricted to `ALLOWED_DOMAINS` (e.g. `webandcrafts.com`).
- **SSRF Prevention**: IP validation blocks private subnet ranges (`10.0.0.0/8`, `192.168.0.0/16`, `127.0.0.0/8`).
- **Canonical Normalization**: Cleans trailing slashes, fragments, and tracking query parameters.
- **Incremental Versioning**: Stores SHA-256 content hashes in `rag_documents`. If a document is updated, the previous active chunks are deactivated and new versioned chunks are written to `rag_chunks`.

---

## 3. Semantic Chunking Strategy

[`SemanticChunker`](file:///c:/WAC/AI%20Chat_Bot/app/rag/chunking/semantic_chunker.py) uses a heading-aware strategy:
- Preserves the structural hierarchy in `heading_path` (e.g., `["Services", "Custom Software", "Backend Engineering"]`).
- Target chunk size: **400–800 tokens** with **50–100 token overlap**.
- Preserves complete sentences and bulleted technical lists.

---

## 4. Query Rewriting & Intent Detection

Before querying the vector and keyword indexes, the user query passes through [`QueryRewriter`](file:///c:/WAC/AI%20Chat_Bot/app/rag/retrieval/query_rewriter.py):

1. **Coreference & History Resolution**:
   Resolves conversational pronouns (e.g., *"What technologies do they use for that?"* after an ecommerce question is rewritten to *"What ecommerce technologies does Web and Crafts use?"*).
2. **Intent Classification**:
   - `COMPANY_TECH`: Queries about technologies WAC uses.
   - `SERVICES`: Queries about services WAC provides.
   - `CAREERS`: Queries regarding jobs, culture, and hiring.
   - `COMPANY_INFO`: Queries regarding offices, leadership, and contact details.
   - `GENERAL`: General WAC informational queries.

---

## 5. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

Candidates are retrieved concurrently via two orthogonal search strategies:

### A. Vector Search (Semantic Match)
- Generates a 768-dimensional query embedding via the configured `RAG_EMBEDDING_PROVIDER`.
- Performs cosine similarity search using MongoDB Atlas Vector Search index.

### B. Keyword Search (Lexical Match)
- Executes MongoDB text search matching exact technology keywords, names, and acronyms across chunk title, heading path, and content.

### C. Reciprocal Rank Fusion (RRF)
Vector and lexical results are merged using RRF ($k=60$):

$$RRF(d) = \frac{w_{\text{vec}}}{60 + r_{\text{vec}}(d)} + \frac{w_{\text{key}}}{60 + r_{\text{key}}(d)}$$

---

## 6. Intent-Aware Fusion Reranking

[`FusionReranker`](file:///c:/WAC/AI%20Chat_Bot/app/rag/retrieval/reranker.py) refines the fused candidates:
1. **Service Page Priority**: Boosts official service pages (`/services/*`) and case studies over general blog posts.
2. **URL Diversity**: Enforces a strict 1-chunk-per-URL constraint in top rank positions to prevent a single long article from monopolizing the context window.
3. **Generic Comparison Penalty**: For company technology questions, penalizes generic comparison blogs that lack company attribution.

---

## 7. Company-Specific Evidence Grounding

[`RAGService._evaluate_evidence_sufficiency`](file:///c:/WAC/AI%20Chat_Bot/app/services/rag_service.py) prevents false technology attribution:

- **The Problem**: A blog titled *"Node.js vs PHP"* discusses Node.js, but does not prove WAC uses Node.js for client development.
- **The Solution**: For `COMPANY_TECH` queries, chunks are required to contain **explicit company relationship markers** (e.g. `"our team"`, `"we provide"`, `"Web and Crafts builds"`, `"our developers"`) or originate from service URLs.
- If only generic ungrounded discussions are found, the gate marks `evidence_sufficient = False`, triggering a polite grounded refusal.

---

## 8. Embedding Provider Decoupling

The embedding subsystem supports two decoupled providers:

```
                  EmbeddingProviderFactory
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
 GeminiEmbeddingProvider            LocalEmbeddingProvider
 (gemini-embedding-001)             (BAAI/bge-base-en-v1.5)
   - Google GenAI API                  - Local CPU/CUDA inference
   - 768 Dimensions                    - 768 Dimensions
   - Subject to API quotas             - Zero API quotas
```

### Provider Independence Matrix
Generation and embedding are completely decoupled via environment variables:

| `RAG_EMBEDDING_PROVIDER` | `DEFAULT_PROVIDER` | Generation Model | Embedding Model | External Calls |
| :--- | :--- | :--- | :--- | :--- |
| `gemini` | `gemini` | Gemini 2.5 Flash | `gemini-embedding-001` | Google GenAI |
| `gemini` | `groq` | Groq Llama 3.3 70B | `gemini-embedding-001` | Groq + Google GenAI |
| `local` | `gemini` | Gemini 2.5 Flash | `bge-base-en-v1.5` | Google GenAI only |
| `local` | `groq` | Groq Llama 3.3 70B | `bge-base-en-v1.5` | **Zero Google Quota** |

---

## 9. Vector Space Isolation & Migration Notice

> [!IMPORTANT]
> **Vector Space Isolation:**
> Vectors generated by `gemini-embedding-001` and `BAAI/bge-base-en-v1.5` exist in mathematically different latent spaces. A query embedding from BGE cannot match chunk embeddings created by Gemini.

- **Current Active Production Provider**: `gemini`
- **Database Status**: Contains Gemini 768d embeddings.
- **Migration Status**: **PAUSED / NOT RUN**.

To switch production embeddings to Local BGE:
```bash
python scripts/migrate_embeddings.py --provider local --dry-run
python scripts/migrate_embeddings.py --provider local --confirm
```
See [docs/embedding-migration.md](file:///c:/WAC/AI%20Chat_Bot/docs/embedding-migration.md) for details.
