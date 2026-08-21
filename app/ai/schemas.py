from dataclasses import dataclass, field
from typing import Optional


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

    usage_source: Optional[str] = None  # provider_metadata, provider_headers, calculated, unavailable
    quota_scope: Optional[str] = None   # request, minute, day, session, account, project, unknown


@dataclass
class AIResponse:
    content: str
    usage: AIUsage