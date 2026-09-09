import asyncio

from app.core.database import connect_db, disconnect_db, get_database


async def main():
    await connect_db()

    db = get_database()
    collection = db.rag_chunks

    total_chunks = await collection.count_documents(
        {"status": "active"}
    )

    pipeline = [
        {"$match": {"status": "active"}},
        {
            "$group": {
                "_id": "$url",
            }
        },
        {"$count": "unique_urls"},
    ]

    result = await collection.aggregate(pipeline).to_list(length=1)

    unique_urls = (
        result[0]["unique_urls"]
        if result
        else 0
    )

    print("=" * 60)
    print("CRAWLER / RAG DATABASE AUDIT")
    print("=" * 60)

    print(f"Active chunks : {total_chunks}")
    print(f"Unique URLs   : {unique_urls}")

    print("\nSample indexed URLs:")

    cursor = collection.find(
        {"status": "active"},
        {
            "_id": 0,
            "url": 1,
            "title": 1,
        },
    ).limit(30)

    async for item in cursor:
        print(
            f"- {item.get('title', '')} | "
            f"{item.get('url', '')}"
        )

    await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())