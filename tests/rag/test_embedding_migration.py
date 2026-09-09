import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from app.core.config import settings
from app.rag.models import RAGChunkModel
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.indexing.indexer import DocumentIndexer
from app.repositories.rag_repository import RAGChunkRepository
from app.services.crawl_service import CrawlService


class FakeCursor:
    """Mock async cursor supporting to_list pagination."""
    def __init__(self, items):
        self.items = list(items)
        self.index = 0

    async def to_list(self, length=None):
        if length is None:
            res = self.items[self.index:]
            self.index = len(self.items)
            return res
        res = self.items[self.index:self.index + length]
        self.index += len(res)
        return res

    def sort(self, key, direction=1):
        return self


@pytest.mark.anyio
async def test_mismatched_chunks_query_filters():
    """Verify repository query filter logic for mismatched chunks."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.rag_chunks = mock_collection
    chunk_repo = RAGChunkRepository(db=mock_db)

    # 1. Test get_mismatched_chunks_count query structure
    mock_collection.count_documents = AsyncMock(return_value=8618)
    count = await chunk_repo.get_mismatched_chunks_count(
        model="gemini-embedding-001",
        dimensions=768,
    )
    assert count == 8618
    mock_collection.count_documents.assert_called_once()
    called_query = mock_collection.count_documents.call_args[0][0]
    assert called_query["status"] == "active"
    assert "$or" in called_query
    or_clauses = called_query["$or"]
    assert {"embedding_model": {"$ne": "gemini-embedding-001"}} in or_clauses
    assert {"embedding_dimensions": {"$ne": 768}} in or_clauses
    assert {"embedding_model": {"$exists": False}} in or_clauses
    assert {"embedding_dimensions": {"$exists": False}} in or_clauses


@pytest.mark.anyio
async def test_reindex_all_skips_when_no_mismatched_chunks():
    """Verify reindex_all exits immediately when 0 mismatched chunks exist."""
    mock_chunk_repo = AsyncMock(spec=RAGChunkRepository)
    mock_chunk_repo.get_mismatched_chunks_count.return_value = 0

    crawl_service = CrawlService(chunk_repo=mock_chunk_repo)
    result = await crawl_service.reindex_all(batch_size=25, force_all=False)

    assert result["chunks_found"] == 0
    assert result["chunks_processed"] == 0
    assert result["chunks_failed"] == 0
    assert result["chunks_remaining"] == 0
    assert mock_chunk_repo.get_mismatched_chunks.call_count == 0


@pytest.mark.anyio
async def test_reindex_all_processes_only_mismatched_chunks_in_batches():
    """Verify reindex_all processes batches of mismatched chunks and updates embeddings."""
    mock_chunk_repo = AsyncMock(spec=RAGChunkRepository)
    mock_indexer = MagicMock(spec=DocumentIndexer)
    mock_embedding_service = AsyncMock(spec=EmbeddingService)
    mock_indexer.embedding_service = mock_embedding_service

    # Prepare 3 mismatched chunks
    chunk_1 = {
        "_id": ObjectId("660000000000000000000001"),
        "content": "Web and crafts builds cloud architectures.",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }
    chunk_2 = {
        "_id": ObjectId("660000000000000000000002"),
        "content": "WAC provides digital transformation and AI.",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }
    chunk_3 = {
        "_id": ObjectId("660000000000000000000003"),
        "content": "WAC ecommerce services using Adobe Commerce.",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }

    mock_chunk_repo.get_mismatched_chunks_count.return_value = 3
    mock_chunk_repo.get_mismatched_chunks.return_value = FakeCursor([chunk_1, chunk_2, chunk_3])
    mock_chunk_repo.update_embedding.return_value = True

    # Return valid 768-dim embeddings
    mock_embedding_service.get_batch_embeddings.side_effect = [
        [[0.1] * 768, [0.2] * 768],  # Batch 1 (2 chunks)
        [[0.3] * 768],                # Batch 2 (1 chunk)
    ]

    crawl_service = CrawlService(
        indexer=mock_indexer,
        chunk_repo=mock_chunk_repo,
    )

    result = await crawl_service.reindex_all(batch_size=2, force_all=False)

    assert result["chunks_found"] == 3
    assert result["chunks_processed"] == 3
    assert result["chunks_failed"] == 0
    assert result["chunks_remaining"] == 0

    assert mock_embedding_service.get_batch_embeddings.call_count == 2
    assert mock_chunk_repo.update_embedding.call_count == 3

    # Verify update_embedding was called with target model and dimensions
    calls = mock_chunk_repo.update_embedding.call_args_list
    assert calls[0].kwargs["chunk_id"] == "660000000000000000000001"
    assert calls[0].kwargs["model"] == settings.RAG_EMBEDDING_MODEL
    assert calls[0].kwargs["dimensions"] == settings.RAG_EMBEDDING_DIMENSIONS
    assert len(calls[0].kwargs["embedding"]) == 768


@pytest.mark.anyio
async def test_reindex_all_resumability():
    """Verify that interrupted runs can be resumed and only process remaining chunks."""
    mock_chunk_repo = AsyncMock(spec=RAGChunkRepository)
    mock_indexer = MagicMock(spec=DocumentIndexer)
    mock_embedding_service = AsyncMock(spec=EmbeddingService)
    mock_indexer.embedding_service = mock_embedding_service

    # Simulating second run after 3000 chunks were already migrated:
    # 5618 chunks remaining
    mock_chunk_repo.get_mismatched_chunks_count.return_value = 5618
    remaining_chunk = {
        "_id": ObjectId("660000000000000000000004"),
        "content": "Remaining chunk content.",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }
    mock_chunk_repo.get_mismatched_chunks.return_value = FakeCursor([remaining_chunk])
    mock_chunk_repo.update_embedding.return_value = True
    mock_embedding_service.get_batch_embeddings.return_value = [[0.5] * 768]

    crawl_service = CrawlService(
        indexer=mock_indexer,
        chunk_repo=mock_chunk_repo,
    )

    result = await crawl_service.reindex_all(batch_size=25, force_all=False)

    assert result["chunks_found"] == 5618
    assert result["chunks_processed"] == 1
    assert result["chunks_remaining"] == 5617


@pytest.mark.anyio
async def test_reindex_all_dimension_mismatch_prevents_db_update():
    """Verify that invalid embedding dimensions trigger error and prevent database corruption."""
    mock_chunk_repo = AsyncMock(spec=RAGChunkRepository)
    mock_indexer = MagicMock(spec=DocumentIndexer)
    mock_embedding_service = AsyncMock(spec=EmbeddingService)
    mock_indexer.embedding_service = mock_embedding_service

    chunk = {
        "_id": ObjectId("660000000000000000000005"),
        "content": "Test content",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }
    mock_chunk_repo.get_mismatched_chunks_count.return_value = 1
    mock_chunk_repo.get_mismatched_chunks.return_value = FakeCursor([chunk])

    # Return invalid dimension embedding (e.g. 1536 instead of 768)
    mock_embedding_service.get_batch_embeddings.return_value = [[0.1] * 1536]

    crawl_service = CrawlService(
        indexer=mock_indexer,
        chunk_repo=mock_chunk_repo,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await crawl_service.reindex_all(batch_size=25, force_all=False)

    assert "Embedding dimension mismatch" in str(exc_info.value)
    # Database update must never have been called
    assert mock_chunk_repo.update_embedding.call_count == 0


@pytest.mark.anyio
async def test_update_embedding_preserves_chunk_metadata():
    """Verify RAGChunkRepository.update_embedding only updates embedding fields via $set."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.rag_chunks = mock_collection
    chunk_repo = RAGChunkRepository(db=mock_db)

    mock_result = MagicMock()
    mock_result.matched_count = 1
    mock_collection.update_one = AsyncMock(return_value=mock_result)

    chunk_id = "660000000000000000000009"
    embedding = [0.1] * 768

    success = await chunk_repo.update_embedding(
        chunk_id=chunk_id,
        embedding=embedding,
        model="gemini-embedding-001",
        dimensions=768,
    )

    assert success is True
    mock_collection.update_one.assert_called_once()
    filter_arg, update_arg = mock_collection.update_one.call_args[0]
    assert filter_arg == {"_id": ObjectId(chunk_id)}

    # Ensure only embedding fields are in $set
    set_fields = update_arg["$set"]
    assert set_fields["embedding"] == embedding
    assert set_fields["embedding_model"] == "gemini-embedding-001"
    assert set_fields["embedding_dimensions"] == 768
    assert "embedding_updated_at" in set_fields

    # Ensure document_id, content, title, url, status, etc. are NOT overwritten
    assert "content" not in set_fields
    assert "title" not in set_fields
    assert "url" not in set_fields
    assert "document_id" not in set_fields
    assert "status" not in set_fields


@pytest.mark.anyio
async def test_extract_retry_delay_from_structured_retry_info():
    """Verify extraction of retry delay from Google RPC RetryInfo in details."""
    service = EmbeddingService()

    # Google RPC style details
    exc = Exception("429 Quota Exceeded")
    exc.code = 429
    exc.status = "RESOURCE_EXHAUSTED"
    exc.details = {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "39.027693442s",
                }
            ],
        }
    }

    assert service.is_rate_limit_error(exc) is True
    delay = service.extract_retry_delay(exc)
    assert delay == pytest.approx(39.027693442, rel=1e-4)


@pytest.mark.anyio
async def test_extract_retry_delay_from_http_headers():
    """Verify extraction of retry delay from HTTP Retry-After header."""
    service = EmbeddingService()

    exc = Exception("429 Too Many Requests")
    exc.code = 429
    mock_resp = MagicMock()
    mock_resp.headers = {"Retry-After": "42"}
    exc.response = mock_resp

    assert service.is_rate_limit_error(exc) is True
    delay = service.extract_retry_delay(exc)
    assert delay == 42.0


@pytest.mark.anyio
async def test_extract_retry_delay_from_message_text():
    """Verify regex fallback extraction of retry delay from error string."""
    service = EmbeddingService()

    exc = Exception(
        "Resource exhausted for metric 'embed_content_free_tier_requests'. Please retry in 39.027693442s."
    )
    assert service.is_rate_limit_error(exc) is True
    delay = service.extract_retry_delay(exc)
    assert delay == pytest.approx(39.027693442, rel=1e-4)


@pytest.mark.anyio
async def test_server_retry_delay_precedence_and_max_bound():
    """Verify server-provided retry delay takes precedence and is bounded by max_retry_delay."""
    service = EmbeddingService(max_retry_delay=50.0)

    # 1. Server requests 20s delay -> sleeps 20.5s (server delay + 0.5s buffer)
    mock_client = MagicMock()
    service._client = mock_client

    rate_limit_exc = Exception("Rate limit. Please retry in 20.0s.")
    rate_limit_exc.code = 429
    rate_limit_exc.status = "RESOURCE_EXHAUSTED"

    mock_resp = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 768
    mock_resp.embeddings = [mock_emb]

    mock_client.models.embed_content.side_effect = [
        rate_limit_exc,
        mock_resp,
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        embeddings = await service.get_batch_embeddings(["Test text"], retries=2)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        mock_sleep.assert_called_once()
        slept_duration = mock_sleep.call_args[0][0]
        assert slept_duration == pytest.approx(20.5, rel=1e-2)

    # 2. Server requests 120s delay -> bounded by max_retry_delay (50.0s)
    rate_limit_long_exc = Exception("Rate limit. Please retry in 120.0s.")
    rate_limit_long_exc.code = 429
    rate_limit_long_exc.status = "RESOURCE_EXHAUSTED"

    mock_client.models.embed_content.side_effect = [
        rate_limit_long_exc,
        mock_resp,
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep_2:
        embeddings = await service.get_batch_embeddings(["Test text"], retries=2)
        assert len(embeddings) == 1
        mock_sleep_2.assert_called_once()
        slept_duration = mock_sleep_2.call_args[0][0]
        assert slept_duration == 50.0


@pytest.mark.anyio
async def test_retry_attempts_are_bounded_and_raise_exception():
    """Verify retries are bounded and raise EmbeddingException upon exhaustion."""
    service = EmbeddingService(default_retries=2, max_retry_delay=10.0)
    mock_client = MagicMock()
    service._client = mock_client

    rate_limit_exc = Exception("429 RESOURCE_EXHAUSTED")
    rate_limit_exc.code = 429
    mock_client.models.embed_content.side_effect = rate_limit_exc

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        from app.rag.exceptions import EmbeddingException
        with pytest.raises(EmbeddingException) as exc_info:
            await service.get_batch_embeddings(["Text 1"], retries=2)

        assert "failed after 3 attempts" in str(exc_info.value)
        # Should have slept 2 times for 3 total attempts (attempt 0, 1, 2)
        assert mock_sleep.call_count == 2


@pytest.mark.anyio
async def test_non_429_errors_use_exponential_backoff():
    """Verify transient non-rate-limit errors use standard exponential backoff."""
    service = EmbeddingService(max_retry_delay=60.0)
    mock_client = MagicMock()
    service._client = mock_client

    transient_exc = ConnectionResetError("Connection reset by peer")

    mock_resp = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 768
    mock_resp.embeddings = [mock_emb]

    mock_client.models.embed_content.side_effect = [
        transient_exc,
        mock_resp,
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        embeddings = await service.get_batch_embeddings(["Text 1"], retries=1)
        assert len(embeddings) == 1
        mock_sleep.assert_called_once_with(2.0)


@pytest.mark.anyio
async def test_reindex_batch_delay_throttling():
    """Verify delay parameter pauses between batches in reindex_all."""
    mock_chunk_repo = AsyncMock(spec=RAGChunkRepository)
    mock_indexer = MagicMock(spec=DocumentIndexer)
    mock_embedding_service = AsyncMock(spec=EmbeddingService)
    mock_indexer.embedding_service = mock_embedding_service

    chunk_1 = {
        "_id": ObjectId("660000000000000000000010"),
        "content": "Batch 1 content",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }
    chunk_2 = {
        "_id": ObjectId("660000000000000000000011"),
        "content": "Batch 2 content",
        "embedding_model": "text-embedding-004",
        "embedding_dimensions": 768,
        "status": "active",
    }

    mock_chunk_repo.get_mismatched_chunks_count.return_value = 2
    mock_chunk_repo.get_mismatched_chunks.return_value = FakeCursor([chunk_1, chunk_2])
    mock_chunk_repo.update_embedding.return_value = True
    mock_embedding_service.get_batch_embeddings.side_effect = [
        [[0.1] * 768],
        [[0.2] * 768],
    ]

    crawl_service = CrawlService(
        indexer=mock_indexer,
        chunk_repo=mock_chunk_repo,
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await crawl_service.reindex_all(
            batch_size=1,
            force_all=False,
            delay=1.5,
        )

        assert result["chunks_found"] == 2
        assert result["chunks_processed"] == 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1.5)

