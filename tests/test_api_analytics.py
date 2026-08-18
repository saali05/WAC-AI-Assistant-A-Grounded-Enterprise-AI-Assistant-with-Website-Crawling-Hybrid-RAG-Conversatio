import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.ai.exceptions import RateLimitException

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_structured_rate_limit_error_handling():
    with patch("app.services.chat_service.ChatService.send_message", side_effect=RateLimitException(
        message="Gemini API limit reached",
        provider="gemini",
        model="gemini-3.6-flash",
        retry_after_seconds=5,
        limit_type="requests_per_day"
    )):
        response = client.post("/chat", json={"provider": "gemini", "message": "hello"})
        assert response.status_code == 429
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "RATE_LIMITED"
        assert data["error"]["provider"] == "gemini"
        assert data["error"]["model"] == "gemini-3.6-flash"
        assert data["error"]["retry_after_seconds"] == 5
        assert data["error"]["limit_type"] == "requests_per_day"
