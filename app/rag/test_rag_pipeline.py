import asyncio

from app.core.database import connect_db, disconnect_db
from app.services.rag_service import RAGService


async def main():

    await connect_db()

    try:

        service = RAGService()

        queries = [
            "What services does WAC provide?",
            "Does WAC offer digital marketing?",
            "What technologies does WAC work with?",
            "What is WAC's ecommerce service?",
        ]

        for query in queries:

            print("\n")
            print("=" * 80)
            print("QUERY:", query)
            print("=" * 80)

            result = await service.get_grounded_context(
                user_message=query,
                conversation_history="",
            )

            print("\nRelevant:", result.is_relevant)
            print("Has Context:", result.has_context)
            print("Retrieval Score:", result.retrieval_score)

            print("\nSOURCES:")

            for source in result.sources:
                print(source)

            print("\nCONTEXT:")
            print(result.context[:3000])

    finally:

        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())