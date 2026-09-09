import asyncio

from app.core.database import connect_db, disconnect_db
from app.rag.retrieval.hybrid_search import HybridSearch


async def main():

    await connect_db()

    try:

        search = HybridSearch()

        query = "What services does WAC provide?"

        print(f"\nQuery: {query}")
        print("=" * 70)

        results = await search.search(
            query=query,
       
        )

        print(f"\nRESULT COUNT: {len(results)}")

        for index, result in enumerate(results, start=1):

            print("\n" + "-" * 70)
            print(f"RESULT {index}")
            print("-" * 70)

            print("Chunk ID:", result.chunk_id)
            print("Title:", result.title)
            print("Score:", result.score)
            print("Vector Score:", result.vector_score)
            print("Keyword Score:", result.keyword_score)
            print("Fusion Score:", result.fusion_score)
            print("URL:", result.url)

            print("\nContent:")
            print(result.content[:500])

    finally:

        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())