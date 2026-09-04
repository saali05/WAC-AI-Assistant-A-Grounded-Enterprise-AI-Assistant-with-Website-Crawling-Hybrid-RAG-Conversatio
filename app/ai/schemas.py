from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


@dataclass
class AIRequest:
    user_message: str
    conversation_history: str
    company_context: str
    system_prompt: str


@dataclass
class AIUsage:
    provider: str
    model: str
    request_type: str = "text"

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    cached_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None

    estimated_cost: Optional[float] = None
    currency: str = "USD"

    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None

    audio_input_seconds: Optional[float] = None
    audio_output_seconds: Optional[float] = None
    live_session_id: Optional[str] = None

    context_limit: Optional[int] = None
    context_remaining: Optional[int] = None

    provider_limit_requests: Optional[int] = None
    provider_remaining_requests: Optional[int] = None

    provider_limit_tokens: Optional[int] = None
    provider_remaining_tokens: Optional[int] = None

    quota_reset_time: Optional[str] = None

    usage_source: Optional[str] = None
    quota_scope: Optional[str] = None


class AIToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class AIResponse(BaseModel):
    content: str = ""
    usage: Optional[AIUsage] = None
    tool_calls: list[AIToolCall] = Field(default_factory=list)

    # Internal Gemini response state.
    # Excluded from serialization.
    raw_response: Any = Field(
        default=None,
        exclude=True,
    )
        
