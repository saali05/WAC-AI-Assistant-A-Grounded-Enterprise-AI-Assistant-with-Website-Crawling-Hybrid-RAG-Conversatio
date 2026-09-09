from dataclasses import asdict
from datetime import datetime
from typing import Any

from app.ai.schemas import AIUsage
from app.core.config import settings
from app.repositories.message_repository import MessageRepository
from app.repositories.usage_repository import UsageRepository


class UsageService:
    """
    Service responsible for recording AI usage and computing session-scoped analytics.
    """

    def __init__(self) -> None:
        self.usage_repository = UsageRepository()
        self.message_repository = MessageRepository()

    async def record_usage(
        self,
        conversation_id: str,
        usage: AIUsage,
        message_id: str | None = None,
    ) -> str:
        """
        Record a single AI execution usage entry.
        """
        data = asdict(usage)
        data["conversation_id"] = conversation_id
        if message_id:
            data["message_id"] = message_id

        return await self.usage_repository.create(data)

    async def get_session_analytics(self, conversation_id: str) -> dict[str, Any]:
        """
        Calculate and return session analytics strictly scoped to conversation_id.
        """
        # Fetch usage records
        records = await self.usage_repository.get_by_conversation(conversation_id)

        # Count total messages in conversation
        messages = await self.message_repository.get_by_conversation(conversation_id)
        message_count = len(messages)

        if not records:
            pricing_tier = getattr(settings, "AI_PRICING_TIER", "free")
            return {
                "conversation_id": conversation_id,
                "session": {
                    "message_count": message_count,
                    "ai_request_count": 0,
                    "started_at": None,
                    "last_activity_at": None,
                },
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "total": 0,
                },
                "context": {
                    "model": settings.GEMINI_MODEL,
                    "limit": 1048576,
                    "remaining": 1048576,
                },
                "quota": {
                    "available": False,
                    "remaining_requests": None,
                    "remaining_tokens": None,
                    "limit_requests": None,
                    "limit_tokens": None,
                    "reset_time": None,
                    "usage_source": "unavailable",
                    "quota_scope": "unknown",
                    "reason": "No requests made in this session yet",
                },
                "cost": {
                    "estimated": 0.0,
                    "currency": "USD",
                    "pricing_tier": pricing_tier,
                },
                "performance": {
                    "average_latency_ms": None,
                    "average_output_tokens_per_second": None,
                    "average_time_to_first_token_ms": None,
                },
                "providers": {},
                "models": {},
                "voice": {
                    "available": False,
                    "model": settings.GEMINI_LIVE_MODEL,
                    "session_count": 0,
                    "audio_input_seconds": None,
                    "audio_output_seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "latency_ms": None,
                    "estimated_cost": 0.0,
                    "live_session_id": None,
                    "reason": "No voice sessions initiated in this conversation.",
                },
                "request_history": [],
            }

        # Calculate session aggregations
        ai_request_count = len(records)
        started_at = records[0].get("created_at")
        last_activity_at = records[-1].get("created_at")

        if isinstance(started_at, datetime):
            started_at = started_at.isoformat()
        if isinstance(last_activity_at, datetime):
            last_activity_at = last_activity_at.isoformat()

        total_input_tokens = sum(r.get("input_tokens") or 0 for r in records)
        total_output_tokens = sum(r.get("output_tokens") or 0 for r in records)
        total_tokens = sum(r.get("total_tokens") or 0 for r in records)
        total_cost = sum(r.get("estimated_cost") or 0.0 for r in records)

        latencies = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

        speeds = [r.get("tokens_per_second") for r in records if r.get("tokens_per_second") is not None]
        avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else None

        ttft_list = [r.get("time_to_first_token_ms") for r in records if r.get("time_to_first_token_ms") is not None]
        avg_ttft = round(sum(ttft_list) / len(ttft_list), 2) if ttft_list else None

        # Breakdown by Provider and Model
        provider_breakdown: dict[str, dict[str, Any]] = {}
        model_breakdown: dict[str, dict[str, Any]] = {}

        latest_quota = None

        for r in records:
            p = r.get("provider") or "unknown"
            m = r.get("model") or "unknown"
            r_type = r.get("request_type", "text")

            # Provider aggregation
            if p not in provider_breakdown:
                provider_breakdown[p] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                }
            provider_breakdown[p]["request_count"] += 1
            provider_breakdown[p]["input_tokens"] += r.get("input_tokens") or 0
            provider_breakdown[p]["output_tokens"] += r.get("output_tokens") or 0
            provider_breakdown[p]["total_tokens"] += r.get("total_tokens") or 0
            provider_breakdown[p]["estimated_cost"] = round(provider_breakdown[p]["estimated_cost"] + (r.get("estimated_cost") or 0.0), 6)

            # Model aggregation
            if m not in model_breakdown:
                model_breakdown[m] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                }
            model_breakdown[m]["request_count"] += 1
            model_breakdown[m]["input_tokens"] += r.get("input_tokens") or 0
            model_breakdown[m]["output_tokens"] += r.get("output_tokens") or 0
            model_breakdown[m]["total_tokens"] += r.get("total_tokens") or 0
            model_breakdown[m]["estimated_cost"] = round(model_breakdown[m]["estimated_cost"] + (r.get("estimated_cost") or 0.0), 6)

            # Check for header rate limit data
            if r.get("provider_remaining_requests") is not None or r.get("provider_remaining_tokens") is not None:
                latest_quota = r

        # Voice aggregation
        voice_records = [
            r for r in records
            if r.get("request_type") == "voice"
            or r.get("model") == settings.GEMINI_LIVE_MODEL
            or r.get("model") == "gemini-3.1-flash-live-preview"
        ]

        if voice_records:
            has_input_audio = any(r.get("audio_input_seconds") is not None for r in voice_records)
            has_output_audio = any(r.get("audio_output_seconds") is not None for r in voice_records)
            has_input_tokens = any(r.get("input_tokens") is not None for r in voice_records)
            has_output_tokens = any(r.get("output_tokens") is not None for r in voice_records)
            voice_latencies = [r.get("latency_ms") for r in voice_records if r.get("latency_ms") is not None]

            voice_audio_in = sum(r.get("audio_input_seconds") or 0.0 for r in voice_records) if has_input_audio else None
            voice_audio_out = sum(r.get("audio_output_seconds") or 0.0 for r in voice_records) if has_output_audio else None
            voice_in_toks = sum(r.get("input_tokens") or 0 for r in voice_records) if has_input_tokens else None
            voice_out_toks = sum(r.get("output_tokens") or 0 for r in voice_records) if has_output_tokens else None
            voice_total_toks = (voice_in_toks or 0) + (voice_out_toks or 0) if (has_input_tokens or has_output_tokens) else None
            voice_cost = sum(r.get("estimated_cost") or 0.0 for r in voice_records)
            voice_avg_latency = round(sum(voice_latencies) / len(voice_latencies), 2) if voice_latencies else None
            latest_live_session_id = next((r.get("live_session_id") for r in reversed(voice_records) if r.get("live_session_id")), None)
            latest_voice_model = next((r.get("model") for r in reversed(voice_records) if r.get("model")), settings.GEMINI_LIVE_MODEL)

            voice_metrics = {
                "available": True,
                "model": latest_voice_model,
                "session_count": len(voice_records),
                "audio_input_seconds": round(voice_audio_in, 2) if voice_audio_in is not None else None,
                "audio_output_seconds": round(voice_audio_out, 2) if voice_audio_out is not None else None,
                "input_tokens": voice_in_toks,
                "output_tokens": voice_out_toks,
                "total_tokens": voice_total_toks,
                "latency_ms": voice_avg_latency,
                "estimated_cost": round(voice_cost, 6),
                "live_session_id": latest_live_session_id,
                "reason": None,
            }
        else:
            voice_metrics = {
                "available": False,
                "model": settings.GEMINI_LIVE_MODEL,
                "session_count": 0,
                "audio_input_seconds": None,
                "audio_output_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_ms": None,
                "estimated_cost": 0.0,
                "live_session_id": None,
                "reason": "No voice sessions initiated in this conversation.",
            }

        # Latest record info for context & quota
        latest_record = records[-1]
        latest_model = latest_record.get("model") or settings.GEMINI_MODEL
        latest_limit = latest_record.get("context_limit") or 1048576
        latest_remaining = latest_record.get("context_remaining")
        if latest_remaining is None and latest_record.get("input_tokens") is not None:
            latest_remaining = max(latest_limit - latest_record["input_tokens"], 0)

        # Quota payload
        if latest_quota:
            quota_payload = {
                "available": True,
                "remaining_requests": latest_quota.get("provider_remaining_requests"),
                "remaining_tokens": latest_quota.get("provider_remaining_tokens"),
                "limit_requests": latest_quota.get("provider_limit_requests"),
                "limit_tokens": latest_quota.get("provider_limit_tokens"),
                "reset_time": latest_quota.get("quota_reset_time"),
                "usage_source": latest_quota.get("usage_source") or "provider_headers",
                "quota_scope": latest_quota.get("quota_scope") or "minute",
                "reason": None,
            }
        else:
            quota_payload = {
                "available": False,
                "remaining_requests": None,
                "remaining_tokens": None,
                "limit_requests": None,
                "limit_tokens": None,
                "reset_time": None,
                "usage_source": latest_record.get("usage_source") or "unavailable",
                "quota_scope": "unknown",
                "reason": f"Provider API ({latest_record.get('provider')}) does not expose current real-time quota remaining in response",
            }

        # Request history
        history = await self.usage_repository.get_request_history(conversation_id, limit=50)

        pricing_tier = getattr(settings, "AI_PRICING_TIER", "free")

        return {
            "conversation_id": conversation_id,
            "session": {
                "message_count": message_count,
                "ai_request_count": ai_request_count,
                "started_at": started_at,
                "last_activity_at": last_activity_at,
            },
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
            },
            "context": {
                "model": latest_model,
                "limit": latest_limit,
                "remaining": latest_remaining,
                "current_prompt_tokens": latest_record.get("input_tokens"),
            },
            "quota": quota_payload,
            "cost": {
                "estimated": round(total_cost, 6),
                "currency": "USD",
                "pricing_tier": pricing_tier,
            },
            "performance": {
                "average_latency_ms": avg_latency,
                "average_output_tokens_per_second": avg_speed,
                "average_time_to_first_token_ms": avg_ttft,
            },
            "providers": provider_breakdown,
            "models": model_breakdown,
            "voice": voice_metrics,
            "request_history": history,
        }
