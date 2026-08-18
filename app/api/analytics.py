from fastapi import APIRouter, HTTPException

from app.services.usage_service import UsageService

router = APIRouter(
    prefix="/conversations",
    tags=["Analytics"],
)


@router.get("/{conversation_id}/analytics")
async def get_conversation_analytics(conversation_id: str):
    """
    Get session-scoped usage analytics for a specific conversation.
    """
    try:
        usage_service = UsageService()
        analytics = await usage_service.get_session_analytics(conversation_id)
        return analytics
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation analytics: {exc}",
        ) from exc
