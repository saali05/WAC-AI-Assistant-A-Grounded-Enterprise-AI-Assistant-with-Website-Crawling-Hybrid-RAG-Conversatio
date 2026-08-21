import asyncio

from app.rag.embeddings.embedding_service import EmbeddingService


async def main():

    service = EmbeddingService()

    vector = await service.get_embedding(
        "Web and Craft provides digital transformation services."
    )

    print("Model:", service.model)
    print("Expected dimensions:", service.dimensions)
    print("Actual dimensions:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    asyncio.run(main())