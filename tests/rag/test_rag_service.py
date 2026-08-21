import pytest
from app.services.rag_service import RAGService


@pytest.mark.anyio
async def test_rag_service_out_of_domain_refusal():
    service = RAGService()
    result = await service.get_grounded_context("What is the capital of France?")

    assert result.is_relevant is False
    assert result.has_context is False
    assert "Web and Craft" in result.refusal_reason


@pytest.mark.anyio
async def test_rag_service_wac_query():
    service = RAGService()
    result = await service.get_grounded_context("What services does WAC provide?")

    assert result.is_relevant is True
