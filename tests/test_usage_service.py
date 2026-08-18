import pytest
import asyncio
from unittest.mock import AsyncMock
from app.services.usage_service import UsageService


@pytest.mark.anyio
async def test_session_analytics_empty_conversation():
    service = UsageService()
    service.usage_repository.get_by_conversation = AsyncMock(return_value=[])
    service.message_repository.get_by_conversation = AsyncMock(return_value=[])

    analytics = await service.get_session_analytics("empty_session_123")

    assert analytics["conversation_id"] == "empty_session_123"
    assert analytics["session"]["message_count"] == 0
    assert analytics["session"]["ai_request_count"] == 0
    assert analytics["tokens"]["total"] == 0
    assert analytics["cost"]["estimated"] == 0.0
    assert analytics["quota"]["available"] is False
    assert analytics["request_history"] == []


@pytest.mark.anyio
async def test_session_analytics_scoped_isolation():
    service = UsageService()

    mock_records_abc = [
        {
            "conversation_id": "ABC",
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "request_type": "text",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "estimated_cost": 0.0,
            "latency_ms": 500,
            "tokens_per_second": 100.0,
            "context_limit": 1048576,
            "context_remaining": 1048476,
            "usage_source": "provider_metadata",
            "quota_scope": "unknown",
            "created_at": "2026-08-18T10:00:00Z",
        }
    ]

    service.usage_repository.get_by_conversation = AsyncMock(side_effect=lambda cid: mock_records_abc if cid == "ABC" else [])
    service.message_repository.get_by_conversation = AsyncMock(side_effect=lambda cid: [{"id": "1"}] if cid == "ABC" else [])
    service.usage_repository.get_request_history = AsyncMock(return_value=[])

    analytics_abc = await service.get_session_analytics("ABC")
    assert analytics_abc["tokens"]["total"] == 150
    assert analytics_abc["session"]["ai_request_count"] == 1

    analytics_xyz = await service.get_session_analytics("XYZ")
    assert analytics_xyz["tokens"]["total"] == 0
    assert analytics_xyz["session"]["ai_request_count"] == 0
