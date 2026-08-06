from app.repositories.message_repository import MessageRepository


class SessionMemoryService:
    """
    Handles conversation memory for the current session.
    """

    MAX_MESSAGES = 8

    @property
    def message_repository(self):
        return MessageRepository()

    async def build_history(
        self,
        conversation_id: str,
    ) -> str:
        """
        Return only the latest messages from the conversation.
        """

        messages = await self.message_repository.get_by_conversation(
            conversation_id
        )

        recent_messages = messages[-self.MAX_MESSAGES :]

        history = []

        for message in recent_messages:

            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            history.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(history)