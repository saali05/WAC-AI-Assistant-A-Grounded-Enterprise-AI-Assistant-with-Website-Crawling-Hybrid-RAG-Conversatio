from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import connect_db, disconnect_db
from app.core.logging import logger
from app.core.exception_handler import ai_exception_handler

from app.api.chat import router as chat_router
from app.api.conversation import router as conversation_router
from app.ai.exceptions import AIException
from app.api.voice import router as voice_router
from app.api.analytics import router as analytics_router
from app.api.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AI Document Chatbot...")

    await connect_db()

    yield

    logger.info("🛑 Shutting down AI Document Chatbot...")

    await disconnect_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(
    AIException,
    ai_exception_handler,
)

app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(voice_router)
app.include_router(analytics_router)
app.include_router(rag_router)



@app.get("/")
async def root():
    return {
        "message": "AI Document Chatbot API"
    }