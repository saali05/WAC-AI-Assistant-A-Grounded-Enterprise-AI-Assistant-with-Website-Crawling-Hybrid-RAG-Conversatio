import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.ai.pricing import calculate_cost, calculate_token_cost, calculate_model_cost, MODEL_PRICING
from app.ai.schemas import AIUsage, AIResponse
from app.services.chat_service import ChatService
from app.services.usage_service import UsageService
from app.langchain.chain import TokenUsageCallbackHandler
from app.rag.models import RAGResult


client = TestClient(app)


# ============================================================
# 1. Native Chat Exactly-Once Usage Recording
# ============================================================

@pytest.mark.anyio
async def test_native_chat_exactly_once_usage_recording():
    """Verify that a native text chat request records exactly one usage entry."""
    service = ChatService()

    mock_ai_response = AIResponse(
        content="WAC uses Python, React, and AWS.",
        usage=AIUsage(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=150,
            output_tokens=30,
            total_tokens=180,
            estimated_cost=0.00002,
        ),
    )
    mock_rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        context="WAC tech stack context",
        sources=[],
        retrieval_score=0.92,
    )

    with patch("app.services.conversation_service.ConversationService.get_or_create", new=AsyncMock(return_value={"id": "conv_native_1", "title": "Tech Stack"})), \
         patch("app.repositories.message_repository.MessageRepository.create", new=AsyncMock(side_effect=["user_msg_1", "asst_msg_1"])), \
         patch("app.services.session_memory_service.SessionMemoryService.build_history", new=AsyncMock(return_value="")), \
         patch("app.ai.service.AIService.chat", new=AsyncMock(return_value=(mock_ai_response, mock_rag_result))), \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock(return_value="usage_rec_1")) as mock_record_usage, \
         patch("app.core.config.settings.USE_LANGCHAIN_PIPELINE", False):

        result = await service.send_message(
            provider="gemini",
            message="What technologies does WAC use?",
            conversation_id="conv_native_1",
        )

        assert result["conversation_id"] == "conv_native_1"
        assert result["rag_used"] is True
        # Usage must be recorded exactly once
        mock_record_usage.assert_awaited_once()
        call_kwargs = mock_record_usage.call_args[1]
        assert call_kwargs["conversation_id"] == "conv_native_1"
        assert call_kwargs["message_id"] == "asst_msg_1"
        usage_arg = call_kwargs["usage"]
        assert usage_arg.input_tokens == 150
        assert usage_arg.output_tokens == 30
        assert usage_arg.total_tokens == 180


# ============================================================
# 2. LangChain Pipeline Exactly-Once Usage Recording
# ============================================================

@pytest.mark.anyio
async def test_langchain_pipeline_exactly_once_usage_recording():
    """Verify that a LangChain chat request records exactly one usage entry."""
    service = ChatService()

    # Mock LangChainResponse
    mock_lc_response = MagicMock()
    mock_lc_response.answer = "WAC uses Shopify and Magento."
    mock_lc_response.sources = [{"title": "Ecommerce", "url": "https://webandcrafts.com/ecom", "heading": "Services", "score": 0.95}]
    mock_lc_response.usage = MagicMock(prompt_tokens=200, completion_tokens=50, total_tokens=250, model_name="gemini-2.5-flash")

    with patch("app.services.conversation_service.ConversationService.get_or_create", new=AsyncMock(return_value={"id": "conv_lc_1", "title": "Ecommerce"})), \
         patch("app.repositories.message_repository.MessageRepository.create", new=AsyncMock(side_effect=["user_msg_1", "asst_msg_1"])), \
         patch("app.repositories.message_repository.MessageRepository.get_by_conversation", new=AsyncMock(return_value=[])), \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock(return_value="usage_rec_lc")) as mock_record_usage, \
         patch("app.core.config.settings.USE_LANGCHAIN_PIPELINE", True), \
         patch("app.langchain.chain.WACLangChainPipeline.ainvoke", new=AsyncMock(return_value=mock_lc_response)):

        result = await service.send_message(
            provider="gemini",
            message="What ecommerce platforms does WAC use?",
            conversation_id="conv_lc_1",
        )

        assert result["conversation_id"] == "conv_lc_1"
        assert result["rag_used"] is True
        mock_record_usage.assert_awaited_once()
        call_kwargs = mock_record_usage.call_args[1]
        assert call_kwargs["conversation_id"] == "conv_lc_1"
        assert call_kwargs["message_id"] == "asst_msg_1"
        usage_arg = call_kwargs["usage"]
        assert usage_arg.input_tokens == 200
        assert usage_arg.output_tokens == 50
        assert usage_arg.total_tokens == 250
        assert usage_arg.estimated_cost is not None


# ============================================================
# 3. Voice Tool Search Records ZERO Usage Records
# ============================================================

def test_voice_tool_search_records_zero_usage():
    """Verify that POST /voice/tool executes retrieval without writing to UsageService."""
    with patch("app.api.voice.WACRetriever.ainvoke", new=AsyncMock(return_value=[])), \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock()) as mock_usage_record:

        response = client.post(
            "/voice/tool",
            json={"name": "search_wac_knowledge", "arguments": {"query": "WAC leadership"}},
        )

    assert response.status_code == 200
    mock_usage_record.assert_not_called()


# ============================================================
# 4. Completed Voice Message Records ONE Usage Record
# ============================================================

def test_completed_voice_message_records_one_usage_record():
    """Verify that POST /voice/message creates exactly one usage entry with voice metrics."""
    with patch("app.services.conversation_service.ConversationService.get_or_create", new=AsyncMock(return_value={"id": "conv_v_1", "title": "Voice Title"})), \
         patch("app.repositories.message_repository.MessageRepository.create", new=AsyncMock(side_effect=["m1", "m2"])), \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock(return_value="rec_1")) as mock_usage_record:

        response = client.post(
            "/voice/message",
            json={
                "conversation_id": "conv_v_1",
                "user_message": "Tell me about WAC",
                "assistant_message": "WAC is a global digital transformation company.",
                "audio_input_seconds": 3.2,
                "audio_output_seconds": 6.8,
                "input_tokens": 90,
                "output_tokens": 45,
                "latency_ms": 320.0,
                "live_session_id": "live_session_123",
            },
        )

    assert response.status_code == 200
    mock_usage_record.assert_awaited_once()
    usage_obj = mock_usage_record.call_args[1]["usage"]
    assert usage_obj.request_type == "voice"
    assert usage_obj.audio_input_seconds == 3.2
    assert usage_obj.audio_output_seconds == 6.8
    assert usage_obj.input_tokens == 90
    assert usage_obj.output_tokens == 45
    assert usage_obj.total_tokens == 135
    assert usage_obj.live_session_id == "live_session_123"


# ============================================================
# 5. Token Calculation & Normalization
# ============================================================

def test_token_cost_calculation():
    """Verify calculate_token_cost and calculate_model_cost mathematical accuracy."""
    # 1,000 input tokens at $0.10/1M = $0.0001
    # 2,000 output tokens at $0.40/1M = $0.0008
    cost = calculate_token_cost(
        input_tokens=1000,
        output_tokens=2000,
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
    )
    assert cost == pytest.approx(0.0009, rel=1e-6)

    # Free tier cost is 0.0
    assert calculate_cost("gemini-2.5-flash", 1000, 2000, pricing_tier="free") == 0.0


# ============================================================
# 6. Unknown Model Pricing Fallback Behavior
# ============================================================

def test_unknown_model_pricing_fallback():
    """Verify calculate_model_cost correctly identifies provider patterns for non-standard model strings."""
    # Custom Llama model maps to Groq pricing
    cost_llama = calculate_model_cost("llama-custom-v1", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_llama > 0.0

    # Custom Gemini model maps to Gemini pricing
    cost_gemini = calculate_model_cost("gemini-custom-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_gemini > 0.0

    # Completely unknown model returns 0.0 without crashing
    cost_unknown = calculate_model_cost("unknown-provider-model-xyz", input_tokens=1000, output_tokens=1000)
    assert cost_unknown == 0.0


# ============================================================
# 7. Session Analytics Aggregation & Isolation
# ============================================================

@pytest.mark.anyio
async def test_session_analytics_aggregation_and_isolation():
    """Verify session analytics accurately sums tokens, costs, and voice metrics per session."""
    service = UsageService()

    records_session_1 = [
        {
            "conversation_id": "sess_1",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "request_type": "text",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "estimated_cost": 0.0001,
            "latency_ms": 300.0,
            "created_at": "2026-09-08T10:00:00Z",
        },
        {
            "conversation_id": "sess_1",
            "provider": "gemini",
            "model": "gemini-3.1-flash-live-preview",
            "request_type": "voice",
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
            "estimated_cost": 0.0004,
            "audio_input_seconds": 10.0,
            "audio_output_seconds": 15.0,
            "latency_ms": 400.0,
            "usage_source": "provider_metadata",
            "created_at": "2026-09-08T10:05:00Z",
        },
    ]

    service.usage_repository.get_by_conversation = AsyncMock(side_effect=lambda cid: records_session_1 if cid == "sess_1" else [])
    service.message_repository.get_by_conversation = AsyncMock(return_value=[{"id": "1"}, {"id": "2"}])
    service.usage_repository.get_request_history = AsyncMock(return_value=[])

    # Session 1 analytics
    analytics_1 = await service.get_session_analytics("sess_1")
    assert analytics_1["session"]["ai_request_count"] == 2
    assert analytics_1["tokens"]["input"] == 300
    assert analytics_1["tokens"]["output"] == 150
    assert analytics_1["tokens"]["total"] == 450
    assert analytics_1["cost"]["estimated"] == pytest.approx(0.0005, rel=1e-5)

    # Voice metrics inside session 1
    voice = analytics_1["voice"]
    assert voice["session_count"] == 1
    assert voice["audio_input_seconds"] == 10.0
    assert voice["audio_output_seconds"] == 15.0
    assert voice["input_tokens"] == 200
    assert voice["output_tokens"] == 100
    assert voice["total_tokens"] == 300

    # Provider breakdown
    assert "gemini" in analytics_1["providers"]
    assert analytics_1["providers"]["gemini"]["request_count"] == 2

    # Isolated Session 2 returns clean empty stats
    analytics_2 = await service.get_session_analytics("sess_2_empty")
    assert analytics_2["session"]["ai_request_count"] == 0
    assert analytics_2["tokens"]["total"] == 0
    assert analytics_2["cost"]["estimated"] == 0.0
    assert analytics_2["voice"]["session_count"] == 0


# ============================================================
# 8. TokenUsageCallbackHandler Accumulation
# ============================================================

@pytest.mark.anyio
async def test_token_usage_callback_handler_accumulation():
    """Verify TokenUsageCallbackHandler safely extracts token metadata without double-counting."""
    handler = TokenUsageCallbackHandler()

    # Simulate generation output from LLM
    mock_response = MagicMock()
    mock_response.llm_output = {
        "token_usage": {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
        "model_name": "gemini-2.5-flash",
    }
    mock_response.generations = []

    await handler.on_llm_end(mock_response)

    assert handler.usage.prompt_tokens == 120
    assert handler.usage.completion_tokens == 40
    assert handler.usage.total_tokens == 160
    assert handler.usage.model_name == "gemini-2.5-flash"
