import asyncio

from app.rag.embeddings.embedding_service import EmbeddingService


async def main():

    service = EmbeddingService()

    text = "What services does WAC provide?"

    print("Generating embedding...")
    print("=" * 70)

    embedding = await service.get_embedding(text)

    print("Embedding generated")
    print("Dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])

    print(
        "All zeros:",
        all(value == 0.0 for value in embedding),
    )


if __name__ == "__main__":
    asyncio.run(main())