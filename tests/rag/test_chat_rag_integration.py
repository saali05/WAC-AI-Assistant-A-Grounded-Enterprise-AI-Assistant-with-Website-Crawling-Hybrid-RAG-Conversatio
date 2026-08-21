import pytest
from unittest.mock import AsyncMock, patch
from app.ai.schemas import AIResponse, AIUsage
from app.ai.service import AIService


@pytest.mark.anyio
async def test_ai_service_rag_refusal():
    ai_service = AIService()

    response, rag_result = await ai_service.chat(
        provider="gemini",
        message="What is the capital of France?",
    )

    assert rag_result.is_relevant is False
    assert "Web and Crafts" in response.content or "Web and Craft" in response.content
    assert rag_result.has_context is False


@pytest.mark.anyio
async def test_ai_service_rag_wac_query():
    ai_service = AIService()

    mock_provider = AsyncMock()
    mock_provider.generate.return_value = AIResponse(
        content="WAC provides custom AI development services.",
        usage=AIUsage(provider="gemini", model="gemini-3.6-flash")
    )

    with patch("app.ai.service.ProviderFactory.get_provider", return_value=mock_provider):
        response, rag_result = await ai_service.chat(
            provider="gemini",
            message="What services does WAC provide?",
        )

    assert rag_result.is_relevant is True
    assert "WAC provides" in response.content
