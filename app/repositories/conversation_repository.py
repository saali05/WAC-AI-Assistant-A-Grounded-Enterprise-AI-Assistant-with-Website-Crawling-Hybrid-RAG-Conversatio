from datetime import datetime, UTC

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from app.core.database import get_database


class ConversationRepository:
    """
    Repository responsible for CRUD operations on conversations.
    """

    def __init__(self) -> None:
        db = get_database()
        self.collection: AsyncIOMotorCollection = db.conversations

    async def create(
        self,
        title: str,
    ) -> str:
        """
        Create a new conversation.

        Args:
            title: Conversation title.

        Returns:
            Newly created conversation ID.
        """

        now = datetime.now(UTC)

        document = {
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

        result = await self.collection.insert_one(document)

        return str(result.inserted_id)

    async def get_by_id(
        self,
        conversation_id: str,
    ) -> dict | None:
        """
        Return a conversation by ID.
        """

        conversation = await self.collection.find_one(
            {
                "_id": ObjectId(conversation_id)
            }
        )

        if conversation is None:
            return None

        conversation["id"] = str(conversation.pop("_id"))

        return conversation

    async def get_all(self) -> list[dict]:
        """
        Return all conversations sorted by most recently updated.
        """

        cursor = self.collection.find().sort(
            "updated_at",
            -1,
        )

        conversations = await cursor.to_list(length=None)

        for conversation in conversations:
            conversation["id"] = str(
                conversation.pop("_id")
            )

        return conversations

    async def rename(
        self,
        conversation_id: str,
        title: str,
    ) -> bool:
        """
        Rename a conversation.
        """

        result = await self.collection.update_one(
            {
                "_id": ObjectId(conversation_id)
            },
            {
                "$set": {
                    "title": title,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

        return result.modified_count > 0

    async def touch(
        self,
        conversation_id: str,
    ) -> None:
        """
        Update conversation timestamp.
        """

        await self.collection.update_one(
            {
                "_id": ObjectId(conversation_id)
            },
            {
                "$set": {
                    "updated_at": datetime.now(UTC)
                }
            },
        )

    async def delete(
        self,
        conversation_id: str,
    ) -> bool:
        """
        Delete a conversation.
        """

        result = await self.collection.delete_one(
            {
                "_id": ObjectId(conversation_id)
            }
        )

        return result.deleted_count > 0