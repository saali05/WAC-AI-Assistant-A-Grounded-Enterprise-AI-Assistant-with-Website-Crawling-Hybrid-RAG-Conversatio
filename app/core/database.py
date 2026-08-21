from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings
from app.core.logging import logger

client: AsyncIOMotorClient | None = None
database: AsyncIOMotorDatabase | None = None

async def connect_db() -> None:
    """
    Create MongoDB connection.
    """
    global client, database

    try:
        client = AsyncIOMotorClient(settings.MONGODB_URI)
        database = client[settings.DATABASE_NAME]

        # Verify the connection
        await client.admin.command("ping")

        logger.info("✅ Connected to MongoDB")

        # Initialize RAG collection indexes
        if settings.RAG_ENABLED:
            from app.repositories.rag_repository import initialize_rag_indexes
            await initialize_rag_indexes(database)

    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise

async def disconnect_db() -> None:
    """
    Close MongoDB connection.
    """
    global client

    if client:
        client.close()
        logger.info("🔌 MongoDB connection closed")

def get_database() -> AsyncIOMotorDatabase:
    """
    Return the active MongoDB database instance.
    """
    if database is None:
        raise RuntimeError("Database connection has not been initialized.")

    return database