from datetime import UTC, datetime
from typing import Any
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING

from app.core.database import get_database


class UsageRepository:
    """
    Repository responsible for AI usage data persistence and aggregation.
    """

    def __init__(self) -> None:
        self._indexes_created = False

    @property
    def collection(self) -> AsyncIOMotorCollection:
        db = get_database()
        return db.ai_usage


    async def ensure_indexes(self) -> None:
        if not self._indexes_created:
            try:
                await self.collection.create_index([("conversation_id", ASCENDING), ("created_at", ASCENDING)])
                await self.collection.create_index([("conversation_id", ASCENDING), ("provider", ASCENDING)])
                await self.collection.create_index([("conversation_id", ASCENDING), ("model", ASCENDING)])
                self._indexes_created = True
            except Exception:
                pass

    async def create(self, usage_data: dict[str, Any]) -> str:
        """
        Store an AI usage document.
        """
        await self.ensure_indexes()

        if "created_at" not in usage_data:
            usage_data["created_at"] = datetime.now(UTC)

        result = await self.collection.insert_one(usage_data)
        return str(result.inserted_id)

    async def get_by_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        """
        Get all usage records for a specific conversation sorted by created_at.
        """
        cursor = self.collection.find({"conversation_id": conversation_id}).sort("created_at", ASCENDING)
        records = await cursor.to_list(length=None)
        for record in records:
            record["id"] = str(record.pop("_id"))
        return records

    async def aggregate_session_usage(self, conversation_id: str) -> dict[str, Any]:
        """
        Aggregate usage stats strictly scoped to conversation_id.
        """
        pipeline = [
            {"$match": {"conversation_id": conversation_id}},
            {
                "$group": {
                    "_id": "$conversation_id",
                    "ai_request_count": {"$sum": 1},
                    "total_input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
                    "total_output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
                    "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
                    "total_cost": {"$sum": {"$ifNull": ["$estimated_cost", 0.0]}},
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                    "avg_tokens_per_second": {"$avg": "$tokens_per_second"},
                    "started_at": {"$min": "$created_at"},
                    "last_activity_at": {"$max": "$created_at"},
                }
            },
        ]

        cursor = self.collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        if result:
            return result[0]
        return {
            "ai_request_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": None,
            "avg_tokens_per_second": None,
            "started_at": None,
            "last_activity_at": None,
        }

    async def get_request_history(self, conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Fetch request history for conversation_id.
        """
        cursor = (
            self.collection.find({"conversation_id": conversation_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        records = await cursor.to_list(length=limit)
        history = []
        for record in records:
            history.append({
                "id": str(record["_id"]),
                "time": record["created_at"].isoformat() if isinstance(record.get("created_at"), datetime) else str(record.get("created_at")),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "request_type": record.get("request_type", "text"),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "total_tokens": record.get("total_tokens"),
                "latency_ms": record.get("latency_ms"),
                "tokens_per_second": record.get("tokens_per_second"),
                "estimated_cost": record.get("estimated_cost"),
                "usage_source": record.get("usage_source"),
                "quota_scope": record.get("quota_scope"),
            })
        return history
