# WAC AI Assistant — Embedding Vector Space Isolation & Migration Guide

---

## 1. Overview & Core Mathematical Principle

The WAC AI Assistant supports two decoupled embedding providers:
1. **Google Gemini Embedding**: `gemini-embedding-001` (768 dimensions)
2. **Local BGE Embedding**: `BAAI/bge-base-en-v1.5` (768 dimensions via `sentence-transformers`)

> [!CAUTION]
> **Dimensional Compatibility ≠ Semantic Interoperability:**
> Although both models produce dense vectors of length `768`, their underlying latent representations, projection weights, and coordinate bases are mathematically orthogonal and completely incompatible.

```
                      Latent Vector Space Comparison
                      
       Gemini Embedding Space                     BGE Embedding Space
     ┌────────────────────────┐                ┌────────────────────────┐
     │  "Custom Software"     │                │  "Custom Software"     │
     │  [+0.41, -0.88, ...]   │                │  [-0.12, +0.65, ...]   │
     │                        │   CANNOT MIX   │                        │
     │  "Ecommerce API"       │ <────────────> │  "Ecommerce API"       │
     │  [+0.39, -0.82, ...]   │                │  [-0.15, +0.61, ...]   │
     └────────────────────────┘                └────────────────────────┘
```

If you query a MongoDB database containing **Gemini chunk vectors** using a **Local BGE query vector**, cosine similarity scores will degenerate into near-random noise, destroying RAG retrieval relevance.

---

## 2. Current Production Status

* **Active Production Embedding Provider**: `gemini` (`gemini-embedding-001`)
* **MongoDB Corpus State**: Indexed entirely in the **Gemini vector space**.
* **Migration Status**: **PAUSED / NOT RUN**.

The Local BGE provider exists as a fully implemented, tested zero-quota alternative, but has intentionally **not** been applied to the live database to preserve existing embeddings.

---

## 3. The Migration Tool (`scripts/migrate_embeddings.py`)

A dedicated migration CLI utility is provided to re-embed all active chunks in MongoDB safely.

### Built-in Safety Controls:
1. **Target Provider Selection (`--provider`)**: Defaults to `local`. Supports `local` or `gemini`.
2. **Dry-Run Mode (`--dry-run`)**: Reads database chunks, initializes the target embedding model, computes a sample embedding, and calculates batch requirements without writing a single byte to MongoDB.
3. **Destructive Guard (`--confirm`)**: Any operation that mutates MongoDB records **strictly requires** the `--confirm` flag. Executing without `--confirm` will automatically abort.

---

## 4. Migration Execution Procedure

If you decide to switch the production database from Gemini to Local BGE embeddings, execute the following steps in sequence:

### Step 1: Pre-Migration Backup
Create a snapshot backup of your MongoDB database:
```bash
mongodump --uri="mongodb://localhost:27017" --db="wac_ai" --collection="rag_chunks" --out="./backup_pre_migration"
```

### Step 2: Perform Dry-Run Validation
Verify database connectivity, chunk count, and model loading without modifying data:
```bash
python scripts/migrate_embeddings.py --provider local --dry-run
```
*Expected Output:*
```text
DRY-RUN MODE: No database records will be modified.
Found N chunks in MongoDB.
Successfully validated local BGE model (768 dimensions).
Dry-run completed successfully.
```

### Step 3: Execute Live Migration
Re-embed the chunk collection with the confirmation flag:
```bash
python scripts/migrate_embeddings.py --provider local --confirm
```
The script will iterate over all active documents in `rag_chunks`, compute new 768-dimensional BGE embeddings in batches using PyTorch `inference_mode()`, and update MongoDB documents.

### Step 4: Update Environment Configuration
In `.env`, switch the active embedding provider:
```ini
RAG_EMBEDDING_PROVIDER="local"
```

### Step 5: Verify Post-Migration RAG Retrieval
Run the RAG verification test suite to ensure the new embeddings retrieve expected WAC content:
```bash
python -m pytest tests/rag/ -q
```

---

## 5. Rollback Procedure

If you need to revert the database back to Gemini embeddings:

### Option A: Re-Migrate to Gemini
```bash
python scripts/migrate_embeddings.py --provider gemini --confirm
```
Then update `.env`:
```ini
RAG_EMBEDDING_PROVIDER="gemini"
```

### Option B: Restore from Backup
```bash
mongorestore --uri="mongodb://localhost:27017" --db="wac_ai" --collection="rag_chunks" ./backup_pre_migration/wac_ai/rag_chunks.bson --drop
```
