import asyncio

from app.core.database import connect_db, disconnect_db
from app.langchain.tools.wac_tools import search_wac_knowledge


async def main():

    await connect_db()

    try:

        result = await search_wac_knowledge.ainvoke(
            {
                "query": "Does WAC offer digital marketing?"
            }
        )

        print("\n")
        print("=" * 80)
        print("LANGCHAIN TOOL RESULT")
        print("=" * 80)
        print(result)

    finally:

        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())