import asyncio

from app.core.database import connect_db, disconnect_db
from app.langchain.retrievers.wac_retriever import WACRetriever


async def main():

    # Initialize MongoDB
    await connect_db()

    try:

        retriever = WACRetriever()

        results = await retriever.ainvoke(
            "What services does WAC provide?"
        )

        print("\nRetrieved documents:")
        print("=" * 60)

        if not results:
            print("No documents retrieved.")
            return

        for index, document in enumerate(results, start=1):

            print(f"\nDocument {index}")
            print("-" * 60)

            print("Content:")
            print(document.page_content[:500])

            print("\nMetadata:")
            print(document.metadata)

    finally:

        # Close MongoDB
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())