from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai

from app.core.config import settings
from app.core.logging import logger

from app.repositories.message_repository import MessageRepository

from typing import Any

from app.services.conversation_service import ConversationService
from app.services.usage_service import UsageService
from app.services.rag_service import RAGService

from app.ai.tools.live_definitions import WAC_LIVE_TOOLS
from app.ai.pricing import calculate_cost, MODEL_PRICING
from app.ai.schemas import AIUsage

from app.langchain.retrievers.wac_retriever import WACRetriever
from app.rag.validation.relevance import WAC_REFUSAL_MESSAGE

router = APIRouter(
    prefix="/voice",
    tags=["Voice"],
)
class VoiceMessageRequest(BaseModel):
    conversation_id: str | None = None
    user_message: str
    assistant_message: str
    audio_input_seconds: float | None = None
    audio_output_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    live_session_id: str | None = None


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

                        "tools": WAC_LIVE_TOOLS,

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
            "tools": WAC_LIVE_TOOLS,
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

        msg_id = await message_repository.create(
            conversation_id=conversation_id,
            role="assistant",
            provider="gemini-live",
            content=request.assistant_message,
        )

        # ----------------------------------------
        # Record Gemini Live Usage
        # ----------------------------------------

        model = settings.GEMINI_LIVE_MODEL
        model_spec = MODEL_PRICING.get(model, {})
        context_limit = model_spec.get("context_limit", 131072)

        input_toks = request.input_tokens
        output_toks = request.output_tokens
        total_toks = (input_toks + output_toks) if (input_toks is not None and output_toks is not None) else None

        cost = calculate_cost(
            model=model,
            input_tokens=input_toks,
            output_tokens=output_toks,
            audio_input_seconds=request.audio_input_seconds,
            audio_output_seconds=request.audio_output_seconds,
        )

        usage = AIUsage(
            provider="gemini",
            model=model,
            request_type="voice",
            input_tokens=input_toks,
            output_tokens=output_toks,
            total_tokens=total_toks,
            estimated_cost=cost,
            currency="USD",
            latency_ms=request.latency_ms,
            audio_input_seconds=request.audio_input_seconds,
            audio_output_seconds=request.audio_output_seconds,
            live_session_id=request.live_session_id,
            context_limit=context_limit,
            context_remaining=(context_limit - input_toks) if input_toks is not None else None,
            usage_source="provider_metadata" if (input_toks is not None or request.audio_input_seconds is not None) else "unavailable",
            quota_scope="unknown",
        )

        usage_service = UsageService()
        await usage_service.record_usage(
            conversation_id=conversation_id,
            usage=usage,
            message_id=msg_id,
        )

        logger.info(
            f"Voice conversation and analytics saved: {conversation_id}"
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

class VoiceToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}
    conversation_id: str | None = None

@router.post("/tool")
async def execute_voice_tool(
    request: VoiceToolRequest,
):
    """
    Execute an approved Gemini Live function call using the LangChain RAG retriever.
    """
    if request.name != "search_wac_knowledge":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice tool: {request.name}",
        )

    query = request.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Tool argument 'query' is required.",
        )

    try:
        # --------------------------------------------------
        # Execute via LangChain WACRetriever
        # --------------------------------------------------
        logger.info("Gemini Live invoking LangChain WACRetriever | query=%s", query)
        retriever = WACRetriever()
        documents = await retriever.ainvoke(query)

        if not documents:
            logger.info("Gemini Live LangChain search yielded no relevant documents.")
            return {
                "success": True,
                "is_relevant": True,
                "has_context": False,
                "retrieval_score": 0.0,
                "answer": (
                    "I couldn't find reliable information about that "
                    "in WAC's current knowledge base."
                ),
                "context": "",
                "sources": [],
            }

        # Format context into clean text chunks for Gemini Live to speak
        formatted_context_parts = []
        sources = []
        top_score = 0.0

        for doc in documents:
            score = float(doc.metadata.get("score") or doc.metadata.get("fusion_score") or 0.0)
            if score > top_score:
                top_score = score

            formatted_context_parts.append(
                f"Topic: {doc.metadata.get('title', '')}\n"
                f"Section: {doc.metadata.get('heading', '')}\n"
                f"Details: {doc.page_content}"
            )
            sources.append({
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "heading": doc.metadata.get("heading", ""),
                "score": score,
            })

        logger.info(
            "Gemini Live LangChain search completed | docs=%d | top_score=%.4f",
            len(documents),
            top_score,
        )

        return {
            "success": True,
            "is_relevant": True,
            "has_context": True,
            "retrieval_score": top_score,
            "context": "\n\n".join(formatted_context_parts),
            "sources": sources,
        }

    except Exception as exc:
        logger.exception("Gemini Live LangChain RAG tool failed")
        raise HTTPException(
            status_code=500,
            detail="WAC knowledge search failed.",
        ) from exc