from fastapi import APIRouter, HTTPException

from app.ai.service import AIService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

ai_service = AIService()


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    try:
        response = await ai_service.chat(
            provider=request.provider,
            message=request.message,
        )

        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )