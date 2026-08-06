from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository


class ConversationService:

    @property
    def conversation_repository(self):
        return ConversationRepository()

    @property
    def message_repository(self):
        return MessageRepository()

    async def get_or_create(
        self,
        conversation_id: str | None,
        first_message: str,
    ) -> dict:
        """
        Return an existing conversation or create a new one.
        """

        if conversation_id:

            conversation = await self.conversation_repository.get_by_id(
                conversation_id
            )

            if conversation:

                await self.conversation_repository.touch(
                    conversation_id
                )

                return conversation

        title = self.generate_title(first_message)

        new_id = await self.conversation_repository.create(
            title=title,
        )

        return await self.conversation_repository.get_by_id(
            new_id
        )

    async def rename(
        self,
        conversation_id: str,
        title: str,
    ) -> bool:

        return await self.conversation_repository.rename(
            conversation_id,
            title,
        )

    async def delete(
        self,
        conversation_id: str,
    ) -> bool:

        await self.message_repository.delete_by_conversation(
            conversation_id
        )

        return await self.conversation_repository.delete(
            conversation_id
        )

    async def get_all(self):

        return await self.conversation_repository.get_all()

    async def get(
        self,
        conversation_id: str,
    ):

        conversation = await self.conversation_repository.get_by_id(
            conversation_id
        )

        if conversation is None:
            return None

        messages = (
            await self.message_repository.get_by_conversation(
                conversation_id
            )
        )

        conversation["messages"] = messages

        return conversation

    @staticmethod
    def generate_title(
        message: str,
    ) -> str:
        """
        Generate a default conversation title.
        """

        message = " ".join(message.split())

        if len(message) <= 50:
            return message

        return message[:50].rstrip() + "..."