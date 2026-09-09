import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import connect_db, disconnect_db, get_database



async def main():
    await connect_db()

    try:
        db = get_database()
        collection = db.rag_chunks

        active_filter = {"status": "active"}

        total = await collection.count_documents({})
        active = await collection.count_documents(active_filter)

        missing_content = await collection.count_documents({
            **active_filter,
            "$or": [
                {"content": {"$exists": False}},
                {"content": ""},
                {"content": None},
            ],
        })

        missing_embedding = await collection.count_documents({
            **active_filter,
            "$or": [
                {"embedding": {"$exists": False}},
                {"embedding": []},
                {"embedding": None},
            ],
        })

        wrong_dimensions = await collection.count_documents({
            **active_filter,
            "embedding_dimensions": {"$ne": 768},
        })

        pipeline = [
            {"$match": active_filter},
            {"$group": {"_id": "$url"}},
            {"$count": "count"},
        ]

        result = await collection.aggregate(
            pipeline
        ).to_list(length=1)

        unique_urls = result[0]["count"] if result else 0

        print("=" * 60)
        print("RAG DATABASE AUDIT")
        print("=" * 60)

        print(f"Total chunks             : {total}")
        print(f"Active chunks            : {active}")
        print(f"Unique URLs              : {unique_urls}")
        print(f"Missing content          : {missing_content}")
        print(f"Missing embedding        : {missing_embedding}")
        print(f"Wrong embedding dims     : {wrong_dimensions}")

        print("\nEmbedding models:")

        pipeline = [
            {"$match": active_filter},
            {
                "$group": {
                    "_id": "$embedding_model",
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]

        async for row in collection.aggregate(pipeline):
            print(
                f"  {row['_id']} : {row['count']}"
            )

        print("\nSample documents:")

        cursor = collection.find(
            active_filter,
            {
                "_id": 0,
                "url": 1,
                "title": 1,
                "heading_path": 1,
                "chunk_index": 1,
                "embedding_model": 1,
                "embedding_dimensions": 1,
            },
        ).limit(20)

        async for row in cursor:
            print(
                f"\nURL: {row.get('url')}"
                f"\nTitle: {row.get('title')}"
                f"\nHeading: {row.get('heading_path')}"
                f"\nChunk: {row.get('chunk_index')}"
                f"\nModel: {row.get('embedding_model')}"
                f"\nDimensions: {row.get('embedding_dimensions')}"
            )

    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())