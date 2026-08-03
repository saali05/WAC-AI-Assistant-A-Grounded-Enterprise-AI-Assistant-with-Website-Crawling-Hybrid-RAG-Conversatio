from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.database import connect_db, disconnect_db
from app.core.logging import logger
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan.
    """
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


app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "message": "AI Document Chatbot API"
    }