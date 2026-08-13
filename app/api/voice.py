from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.repositories.message_repository import MessageRepository
from app.services.conversation_service import ConversationService


router = APIRouter(
    prefix="/voice",
    tags=["Voice"],
)


class VoiceMessageRequest(BaseModel):
    conversation_id: str | None = None
    user_message: str
    assistant_message: str


@router.get("/token")
async def create_live_token():

    try:

        client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        now = datetime.now(timezone.utc)

        token = client.auth_tokens.create(
            config={
                "uses": 1,

                "expire_time": (
                    now + timedelta(minutes=30)
                ),

                "new_session_expire_time": (
                    now + timedelta(minutes=1)
                ),

                "live_connect_constraints": {
                    "model": settings.GEMINI_LIVE_MODEL,

                    "config": {
                        "response_modalities": [
                            "AUDIO"
                        ],

                        "input_audio_transcription": {},

                        "output_audio_transcription": {},

                        "system_instruction": {
                            "parts": [
                                {
                                    "text": """
You are WAC AI, the official AI assistant
for Web and Craft.

Your ONLY purpose is to answer questions
related to Web and Craft.

You may answer questions about:

- Web and Craft
- WAC services
- WAC technologies
- WAC AI and ML capabilities
- WAC software engineering
- WAC cloud and DevOps
- WAC digital marketing
- WAC branding
- WAC industries
- WAC clients
- WAC case studies
- WAC careers
- WAC leadership
- WAC offices and contact information

STRICT RULE:

If the user's question is unrelated to Web
and Craft, do not answer the question.

Instead say:

"I'm WAC AI, a specialized AI assistant for
Web and Craft. I can only help with questions
related to Web and Craft, its services,
technologies, projects, careers, and company
information."

Never answer unrelated general knowledge questions.

Never invent information about WAC.

VOICE STYLE:

Speak naturally and concisely.

Do not use Markdown.
Do not use hashtags.
Do not use bullet points.
Do not use unnecessary headings.

You are a WAC-specific AI assistant.
"""
                                }
                            ]
                        }
                    }
                }
            }
        )

        logger.info(
            "Gemini Live ephemeral token created"
        )

        return {
            "token": token.name,
            "model": settings.GEMINI_LIVE_MODEL,
        }

    except Exception:

        logger.exception(
            "Failed to create Gemini Live token"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Voice service is temporarily unavailable."
            ),
        )


@router.post("/message")
async def save_voice_message(
    request: VoiceMessageRequest,
):

    try:

        conversation_service = ConversationService()

        message_repository = MessageRepository()

        # ----------------------------------------
        # Get or create session
        # ----------------------------------------

        conversation = (
            await conversation_service.get_or_create(
                conversation_id=request.conversation_id,
                first_message=request.user_message,
            )
        )

        conversation_id = conversation["id"]

        # ----------------------------------------
        # Save user message
        # ----------------------------------------

        await message_repository.create(
            conversation_id=conversation_id,
            role="user",
            provider="gemini-live",
            content=request.user_message,
        )

        # ----------------------------------------
        # Save assistant message
        # ----------------------------------------

        await message_repository.create(
            conversation_id=conversation_id,
            role="assistant",
            provider="gemini-live",
            content=request.assistant_message,
        )

        logger.info(
            f"Voice conversation saved: {conversation_id}"
        )

        return {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "response": request.assistant_message,
        }

    except Exception as exc:

        logger.exception(
            "Failed to save voice conversation"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save voice conversation.",
        ) from exc