from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorCollection

from app.core.database import get_database

class MessageRepository:
    """
    Repository responsible for chat message persistence.
    """

    def __init__(self) -> None:
        db = get_database()
        self.collection: AsyncIOMotorCollection = db.messages

    async def create(
        self,
        conversation_id: str,
        role: str,
        provider: str,
        content: str,
    ) -> str:
        """
        Store a chat message.

        Args:
            conversation_id: Conversation ID.
            role: user | assistant
            provider: gemini | groq
            content: Message content.

        Returns:
            Message ID.
        """

        document = {
            "conversation_id": conversation_id,
            "role": role,
            "provider": provider,
            "content": content,
            "created_at": datetime.now(UTC),
        }

        result = await self.collection.insert_one(document)

        return str(result.inserted_id)

    async def get_by_conversation(
        self,
        conversation_id: str,
    ) -> list[dict]:
        """
        Return all messages in chronological order.
        """

        cursor = (
            self.collection
            .find({"conversation_id": conversation_id})
            .sort("created_at", 1)
        )

        messages = await cursor.to_list(length=None)

        for message in messages:
            message["id"] = str(message.pop("_id"))

        return messages

    async def delete_by_conversation(
        self,
        conversation_id: str,
    ) -> int:
        """
        Delete all messages belonging to a conversation.

        Returns:
            Number of deleted messages.
        """

        result = await self.collection.delete_many(
            {"conversation_id": conversation_id}
        )

        return result.deleted_count