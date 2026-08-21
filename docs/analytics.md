# Session Analytics & Usage Dashboard Documentation

## Overview

The Session Analytics & Usage Dashboard provides real-time, session-scoped performance, token usage, cost estimation, and API quota tracking for the WAC AI Assistant platform.

## Key Design Principles

1. **Strict Session Isolation**: Analytics are filtered strictly by `conversation_id`. Opening a new chat or switching sessions resets displayed counters to zero.
2. **Zero Fabrication**: Provider API quotas are displayed ONLY when explicitly exposed by provider APIs (e.g. Groq `x-ratelimit-*` response headers). Where quota information is unexposed by the provider (e.g., Gemini completion payloads), the dashboard explicitly reports `"Not available from provider API"`.
3. **Dual Progress Bar Tracking**:
   - **Model Context Capacity (Current Request Prompt)**: Displays current prompt tokens against the model's max context limit (e.g., `17,088 / 1,048,576 tokens`). Computed as `context_limit - input_tokens` for the current prompt (which includes system prompt + WAC knowledge + conversation history + user message). Labeled as `"Current Prompt Context Capacity Remaining"`.
   - **Session Total Tokens Consumed**: Displays cumulative token consumption across the current session (e.g., `42,310 tokens consumed`).
4. **Dynamic `quota_scope` & `usage_source` Auditing**:
   - `quota_scope` is derived dynamically from header reset times (`minute`, `day`, or `unknown`).
   - `usage_source`: `provider_metadata` | `provider_headers` | `calculated` | `unavailable`.
5. **Gemini Live Voice Usage**:
   - Recorded only when official provider usage metadata is attached (`usage_source = "provider_metadata"`).
   - If Live API metadata is unavailable, the UI explicitly displays `"Usage data unavailable"` rather than fabricating estimates from microphone timers.

---

## Supported AI Models & Specifications

| Model | Role | Context Window | Max Output | Official Pricing (Free Tier) | Official Pricing (Paid Tier) |
|---|---|---|---|---|---|
| `gemini-3.6-flash` | Normal Text Chatbot | 1,048,576 tokens | 65,536 tokens | $0.00 / 1M | Input: $1.50/1M, Output: $7.50/1M |
| `gemini-3.1-flash-live-preview` | Gemini Live Voice | 131,072 tokens | 65,536 tokens | $0.00 / 1M | Text In: $0.75/1M, Audio In: $3.00/1M, Text Out: $4.50/1M, Audio Out: $12.00/1M |
| `llama-3.3-70b-versatile` | Groq Text Chatbot | 131,072 tokens | 32,768 tokens | Input: $0.59/1M, Output: $0.79/1M | Input: $0.59/1M, Output: $0.79/1M |

### Pricing Verification Sources

- Google Gemini Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Google Gemini Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Google Gemini 3.6 Flash Model: https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash
- Google Gemini Live API & Pricing: https://ai.google.dev/gemini-api/docs/live-api/best-practices
- Groq Llama 3.3 70B & Rate Limits: https://console.groq.com/docs/rate-limits

---

## Data Model (`ai_usage` Collection)

Each AI invocation produces a document in MongoDB `ai_usage` collection:

```json
{
  "conversation_id": "ABC-123",
  "message_id": "64a...",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "request_type": "text",
  "input_tokens": 820,
  "output_tokens": 120,
  "total_tokens": 940,
  "cached_tokens": null,
  "thinking_tokens": null,
  "estimated_cost": 0.000578,
  "currency": "USD",
  "latency_ms": 420.5,
  "tokens_per_second": 285.3,
  "time_to_first_token_ms": null,
  "context_limit": 131072,
  "context_remaining": 130252,
  "provider_limit_requests": 30,
  "provider_remaining_requests": 29,
  "provider_limit_tokens": 12000,
  "provider_remaining_tokens": 11180,
  "quota_reset_time": "6s",
  "usage_source": "provider_headers",
  "quota_scope": "minute",
  "created_at": "2026-08-18T10:00:00Z"
}
```

### MongoDB Indexes
- `conversation_id` + `created_at`
- `conversation_id` + `provider`
- `conversation_id` + `model`
