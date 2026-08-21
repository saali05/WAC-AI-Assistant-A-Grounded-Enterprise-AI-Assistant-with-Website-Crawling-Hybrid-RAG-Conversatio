from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.ai.exceptions import AIException

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):
    chat_service = ChatService()
    try:

        result = await chat_service.send_message(
            provider=request.provider,
            message=request.message,
            conversation_id=request.conversation_id,
        )

        return ChatResponse(**result)

    except AIException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )