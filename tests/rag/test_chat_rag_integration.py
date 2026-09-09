import pytest
from unittest.mock import AsyncMock, patch
from app.ai.schemas import AIResponse, AIUsage
from app.ai.service import AIService
from app.rag.models import RAGResult


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
async def test_ai_service_greeting():
    ai_service = AIService()

    for greeting_input in ["hii", "hello", "hey", "good morning"]:
        response, rag_result = await ai_service.chat(
            provider="gemini",
            message=greeting_input,
        )

        assert rag_result.is_relevant is True
        assert rag_result.has_context is False
        assert "Welcome to Web and Crafts" in response.content or "Web and Crafts" in response.content


@pytest.mark.anyio
async def test_photosynthesis_refusal():
    ai_service = AIService()

    response, rag_result = await ai_service.chat(
        provider="gemini",
        message="what is photosynthesis?",
    )

    assert rag_result.is_relevant is False
    assert "WAC AI Assistant" in response.content or "specifically designed" in response.content


@pytest.mark.anyio
async def test_multiturn_affirmative_followup():
    ai_service = AIService()

    history = (
        "User: What UI/UX and web design solutions does Webandcrafts specialize in?\n"
        "Assistant: Webandcrafts specializes in enterprise UI/UX design, web application development, and e-commerce solutions. "
        "Would you like to discuss how these solutions can be tailored for your specific industry?"
    )

    mock_provider = AsyncMock()
    mock_provider.generate.return_value = AIResponse(
        content="Here is how Webandcrafts tailors UI/UX solutions for healthcare, e-commerce, and fintech industries.",
        usage=AIUsage(provider="gemini", model="gemini-2.5-flash")
    )

    mock_rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        context="Webandcrafts provides industry-specific UI/UX design for healthcare, e-commerce, and fintech.",
        sources=[],
        retrieval_score=0.91
    )

    with patch("app.ai.service.ProviderFactory.get_provider", return_value=mock_provider), \
         patch("app.services.rag_service.RAGService.get_grounded_context", AsyncMock(return_value=mock_rag_result)):
        response, rag_result = await ai_service.chat(
            provider="gemini",
            message="yes i would like to discuss",
            history=history
        )

    # Turn 2 must NOT be marked as out-of-domain refusal
    assert rag_result.is_relevant is True
    assert "Webandcrafts tailors UI/UX" in response.content

    # Turn 3: Off-topic query with history must still return refusal
    response_turn3, rag_result_turn3 = await ai_service.chat(
        provider="gemini",
        message="what is photosynthesis?",
        history=history
    )
    assert rag_result_turn3.is_relevant is False
    assert "specifically designed" in response_turn3.content or "WAC AI Assistant" in response_turn3.content


@pytest.mark.anyio
async def test_ai_service_rag_wac_query():
    ai_service = AIService()

    mock_provider = AsyncMock()
    mock_provider.generate.return_value = AIResponse(
        content="WAC provides custom AI development services.",
        usage=AIUsage(provider="gemini", model="gemini-3.6-flash")
    )

    mock_rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        context="WAC provides custom AI development services.",
        sources=[],
        retrieval_score=0.95
    )

    with patch("app.ai.service.ProviderFactory.get_provider", return_value=mock_provider), \
         patch("app.services.rag_service.RAGService.get_grounded_context", AsyncMock(return_value=mock_rag_result)):
        response, rag_result = await ai_service.chat(
            provider="gemini",
            message="What services does WAC provide?",
        )

    assert rag_result.is_relevant is True
    assert "WAC provides" in response.content
