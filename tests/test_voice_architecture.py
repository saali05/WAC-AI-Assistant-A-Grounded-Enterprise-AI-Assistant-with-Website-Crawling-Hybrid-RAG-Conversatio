import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.ai.tools.live_definitions import WAC_LIVE_TOOLS
from app.prompts.voice_system import VOICE_SYSTEM_PROMPT


client = TestClient(app)


# ============================================================
# 1. GET /voice/token
# ============================================================

def test_voice_token_generation_and_system_prompt():
    """
    Verify GET /voice/token:
    - Generates Live token using Google GenAI client
    - Sets system instruction to the centralized VOICE_SYSTEM_PROMPT
    - Registers WAC_LIVE_TOOLS (including search_wac_knowledge)
    - Returns token name and model configuration
    """
    mock_token = MagicMock()
    mock_token.name = "auth_tokens/test_ephemeral_token_123"

    mock_genai_client = MagicMock()
    mock_genai_client.auth_tokens.create.return_value = mock_token

    with patch("app.api.voice.genai.Client", return_value=mock_genai_client):
        response = client.get("/voice/token")

    assert response.status_code == 200
    data = response.json()
    assert data["token"] == "auth_tokens/test_ephemeral_token_123"
    assert "tools" in data

    # Verify Google GenAI client call configuration
    mock_genai_client.auth_tokens.create.assert_called_once()
    call_kwargs = mock_genai_client.auth_tokens.create.call_args[1]
    config = call_kwargs.get("config", {})

    live_constraints = config.get("live_connect_constraints", {})
    live_config = live_constraints.get("config", {})

    # Verify system instruction matches centralized VOICE_SYSTEM_PROMPT
    sys_instruction = live_config.get("system_instruction", {})
    parts = sys_instruction.get("parts", [])
    assert len(parts) == 1
    assert parts[0]["text"] == VOICE_SYSTEM_PROMPT

    # Verify tools registered
    assert live_config.get("tools") == WAC_LIVE_TOOLS
    assert any(
        func["name"] == "search_wac_knowledge"
        for group in WAC_LIVE_TOOLS
        for func in group.get("function_declarations", [])
    )


# ============================================================
# 2. POST /voice/tool — Valid WAC Query & Source Propagation
# ============================================================

def test_voice_tool_execution_valid_wac_query():
    """
    Verify POST /voice/tool with valid query:
    - Calls LangChain WACRetriever
    - Formats grounded context with Topic, Section, Details
    - Propagates full source citations (title, url, heading, score)
    - Returns success=True, is_relevant=True, has_context=True
    """
    mock_docs = [
        Document(
            page_content="Webandcrafts provides AI software engineering and enterprise commerce.",
            metadata={
                "title": "WAC Enterprise AI",
                "url": "https://webandcrafts.com/services/ai",
                "heading": "Services > AI",
                "score": 0.94,
            },
        ),
        Document(
            page_content="WAC utilizes Adobe Commerce, Shopify, and custom Laravel solutions.",
            metadata={
                "title": "WAC Commerce Solutions",
                "url": "https://webandcrafts.com/services/ecommerce",
                "heading": "Services > Commerce",
                "score": 0.88,
            },
        ),
    ]

    with patch("app.api.voice.WACRetriever.ainvoke", new=AsyncMock(return_value=mock_docs)):
        response = client.post(
            "/voice/tool",
            json={
                "name": "search_wac_knowledge",
                "arguments": {"query": "What ecommerce technologies does WAC use?"},
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["is_relevant"] is True
    assert data["has_context"] is True
    assert data["retrieval_score"] == 0.94
    assert "Topic: WAC Enterprise AI" in data["context"]
    assert "Details: Webandcrafts provides AI software" in data["context"]

    # Verify source propagation
    assert len(data["sources"]) == 2
    assert data["sources"][0]["title"] == "WAC Enterprise AI"
    assert data["sources"][0]["url"] == "https://webandcrafts.com/services/ai"
    assert data["sources"][0]["heading"] == "Services > AI"
    assert data["sources"][0]["score"] == 0.94

    assert data["sources"][1]["title"] == "WAC Commerce Solutions"
    assert data["sources"][1]["url"] == "https://webandcrafts.com/services/ecommerce"
    assert data["sources"][1]["score"] == 0.88


# ============================================================
# 3. POST /voice/tool — Non-WAC Query Refusal
# ============================================================

def test_voice_tool_execution_non_wac_query_refusal():
    """
    Verify POST /voice/tool with non-WAC query:
    - Returns has_context=False and standardized refusal message
    - Context is empty string
    - Sources list is empty
    """
    with patch("app.api.voice.WACRetriever.ainvoke", new=AsyncMock(return_value=[])):
        response = client.post(
            "/voice/tool",
            json={
                "name": "search_wac_knowledge",
                "arguments": {"query": "What is the capital of France?"},
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["is_relevant"] is True
    assert data["has_context"] is False
    assert data["retrieval_score"] == 0.0
    assert data["context"] == ""
    assert data["sources"] == []
    assert "couldn't find reliable information" in data["answer"]


# ============================================================
# 4. POST /voice/tool — Insufficient Evidence
# ============================================================

def test_voice_tool_execution_insufficient_evidence():
    """
    Verify POST /voice/tool when evidence is weak/insufficient:
    - Returns has_context=False
    - Does not fabricate context or sources
    """
    with patch("app.api.voice.WACRetriever.ainvoke", new=AsyncMock(return_value=[])):
        response = client.post(
            "/voice/tool",
            json={
                "name": "search_wac_knowledge",
                "arguments": {"query": "Does WAC make airplanes?"},
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["has_context"] is False
    assert data["context"] == ""
    assert data["sources"] == []


# ============================================================
# 5. POST /voice/tool — Empty / Invalid Query Validation
# ============================================================

def test_voice_tool_validation_errors():
    """
    Verify POST /voice/tool input validation:
    - Empty query -> 400 Bad Request
    - Missing query argument -> 400 Bad Request
    - Unsupported tool name -> 400 Bad Request
    """
    # Empty query string
    r1 = client.post(
        "/voice/tool",
        json={"name": "search_wac_knowledge", "arguments": {"query": "  "}},
    )
    assert r1.status_code == 400
    assert "required" in r1.json()["detail"].lower()

    # Missing query key
    r2 = client.post(
        "/voice/tool",
        json={"name": "search_wac_knowledge", "arguments": {}},
    )
    assert r2.status_code == 400

    # Unknown tool name
    r3 = client.post(
        "/voice/tool",
        json={"name": "unknown_tool", "arguments": {"query": "test"}},
    )
    assert r3.status_code == 400
    assert "Unsupported voice tool" in r3.json()["detail"]


# ============================================================
# 6. POST /voice/message & Usage Accounting
# ============================================================

def test_voice_message_persistence_and_usage_recording():
    """
    Verify POST /voice/message:
    - Creates or retrieves conversation session
    - Saves user and assistant messages with provider='gemini-live'
    - Calculates cost using audio input/output seconds and token metrics
    - Records single AIUsage entry via UsageService
    """
    mock_conv = {"id": "conv_voice_123", "title": "WAC Ecommerce Query"}

    with patch("app.services.conversation_service.ConversationService.get_or_create", new=AsyncMock(return_value=mock_conv)), \
         patch("app.repositories.message_repository.MessageRepository.create", new=AsyncMock(side_effect=["msg_user_1", "msg_asst_1"])) as mock_msg_create, \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock(return_value="usage_rec_1")) as mock_usage_record:

        response = client.post(
            "/voice/message",
            json={
                "conversation_id": "conv_voice_123",
                "user_message": "What ecommerce technologies does WAC use?",
                "assistant_message": "WAC specializes in Adobe Commerce, Shopify, and WooCommerce.",
                "audio_input_seconds": 4.5,
                "audio_output_seconds": 5.2,
                "input_tokens": 120,
                "output_tokens": 40,
                "latency_ms": 350.0,
                "live_session_id": "live_session_xyz_789",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "conv_voice_123"
    assert data["title"] == "WAC Ecommerce Query"

    # Verify user message and assistant message saved with provider="gemini-live"
    assert mock_msg_create.call_count == 2
    user_call = mock_msg_create.call_args_list[0][1]
    assert user_call["role"] == "user"
    assert user_call["provider"] == "gemini-live"
    assert user_call["content"] == "What ecommerce technologies does WAC use?"

    asst_call = mock_msg_create.call_args_list[1][1]
    assert asst_call["role"] == "assistant"
    assert asst_call["provider"] == "gemini-live"
    assert asst_call["content"] == "WAC specializes in Adobe Commerce, Shopify, and WooCommerce."

    # Verify UsageService was called exactly once with request_type="voice"
    mock_usage_record.assert_awaited_once()
    usage_call_kwargs = mock_usage_record.call_args[1]
    usage_obj = usage_call_kwargs["usage"]
    assert usage_obj.request_type == "voice"
    assert usage_obj.provider == "gemini"
    assert usage_obj.audio_input_seconds == 4.5
    assert usage_obj.audio_output_seconds == 5.2
    assert usage_obj.input_tokens == 120
    assert usage_obj.output_tokens == 40
    assert usage_obj.total_tokens == 160
    assert usage_obj.live_session_id == "live_session_xyz_789"
    assert usage_obj.estimated_cost >= 0.0


# ============================================================
# 7. No Duplicate Usage Accounting
# ============================================================

def test_voice_tool_does_not_record_usage():
    """
    Verify POST /voice/tool does NOT invoke UsageService.record_usage
    to prevent double-counting of voice interactions.
    """
    mock_docs = [
        Document(
            page_content="WAC provides cloud engineering.",
            metadata={"title": "Cloud", "url": "https://webandcrafts.com/cloud", "heading": "Cloud", "score": 0.9},
        )
    ]

    with patch("app.api.voice.WACRetriever.ainvoke", new=AsyncMock(return_value=mock_docs)), \
         patch("app.services.usage_service.UsageService.record_usage", new=AsyncMock()) as mock_usage_record:

        response = client.post(
            "/voice/tool",
            json={
                "name": "search_wac_knowledge",
                "arguments": {"query": "What cloud services does WAC provide?"},
            },
        )

    assert response.status_code == 200
    mock_usage_record.assert_not_called()
