from fastapi import APIRouter, HTTPException

from app.ai.chatbot import ChatBotService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

chatbot = ChatBotService()


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    """
    Chat with Gemini AI.
    """
    try:

        response = await chatbot.chat(request.message)

        return ChatResponse(
            response=response
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )