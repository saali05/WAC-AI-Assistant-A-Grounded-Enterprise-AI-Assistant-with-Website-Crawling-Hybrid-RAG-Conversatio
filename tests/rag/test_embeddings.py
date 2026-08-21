import pytest
from app.rag.embeddings.embedding_service import EmbeddingService


@pytest.mark.anyio
async def test_embedding_service_fallback_vector():
    service = EmbeddingService(dimensions=768)
    vec = await service.get_embedding("WAC AI Chatbot Test")
    assert len(vec) == 768
    assert isinstance(vec[0], float)

    batch_vecs = await service.get_batch_embeddings(["Text 1", "Text 2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768
