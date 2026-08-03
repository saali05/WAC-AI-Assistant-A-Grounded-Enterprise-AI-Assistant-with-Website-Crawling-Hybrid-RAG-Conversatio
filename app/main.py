from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.database import connect_db, disconnect_db
from app.core.logging import logger


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


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")

    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
    }