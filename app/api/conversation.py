from fastapi import APIRouter, HTTPException

from app.schemas.conversation import RenameConversationRequest
from app.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# @router.get("")
# async def get_conversations():
#     """
#     Get all conversations.
#     """
#     conversation_service = ConversationService()

#     return await conversation_service.get_all()


# @router.get("/{conversation_id}")
# async def get_conversation(conversation_id: str):
#     """
#     Get a conversation with all its messages.
#     """
#     conversation_service = ConversationService()

#     conversation = await conversation_service.get(conversation_id)

#     if conversation is None:
#         raise HTTPException(
#             status_code=404,
#             detail="Conversation not found",
#         )

#     return conversation


# @router.patch("/{conversation_id}")
# async def rename_conversation(
#     conversation_id: str,
#     request: RenameConversationRequest,
# ):
#     """
#     Rename a conversation.
#     """
#     conversation_service = ConversationService()

#     success = await conversation_service.rename(
#         conversation_id=conversation_id,
#         title=request.title,
#     )

#     if not success:
#         raise HTTPException(
#             status_code=404,
#             detail="Conversation not found",
#         )

#     return {
#         "message": "Conversation renamed successfully"
#     }


# @router.delete("/{conversation_id}")
# async def delete_conversation(conversation_id: str):
#     """
#     Delete a conversation and all its messages.
#     """
#     conversation_service = ConversationService()

#     success = await conversation_service.delete(
#         conversation_id
#     )

#     if not success:
#         raise HTTPException(
#             status_code=404,
#             detail="Conversation not found",
#         )

#     return {
#         "message": "Conversation deleted successfully"
#     }