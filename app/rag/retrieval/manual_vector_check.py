import asyncio

from app.core.database import connect_db, disconnect_db
from app.rag.retrieval.vector_search import VectorSearch


async def main():

    await connect_db()

    try:
        search = VectorSearch()

        query = "What services does WAC provide?"

        print(f"\nQuery: {query}")
        print("=" * 70)

        results = await search.search(
            query=query,
            top_k=5,
        )

        print(f"\nVECTOR RESULT COUNT: {len(results)}")

        for index, result in enumerate(results, start=1):

            print("\n" + "-" * 70)
            print(f"VECTOR RESULT {index}")
            print("-" * 70)

            print("ID:", result.get("id"))
            print("Title:", result.get("title"))
            print("Score:", result.get("score"))
            print("URL:", result.get("url"))

            print("\nContent:")
            print(result.get("content", "")[:500])

    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())