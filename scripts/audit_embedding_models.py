import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import connect_db, disconnect_db, get_database



async def main():
    await connect_db()

    try:
        db = get_database()
        collection = db.rag_chunks

        print("=" * 70)
        print("EMBEDDING MODEL AUDIT")
        print("=" * 70)

        pipeline = [
            {
                "$group": {
                    "_id": {
                        "model": "$embedding_model",
                        "dimensions": "$embedding_dimensions",
                    },
                    "count": {"$sum": 1},
                }
            },
            {
                "$sort": {
                    "count": -1
                }
            },
        ]

        results = await collection.aggregate(
            pipeline
        ).to_list(length=100)

        print("\nStored embedding configurations:\n")

        for item in results:
            config = item["_id"]

            print(
                f"Model      : {config.get('model')}"
            )
            print(
                f"Dimensions : {config.get('dimensions')}"
            )
            print(
                f"Chunks     : {item['count']}"
            )
            print("-" * 50)

        print("\nCurrent application configuration:")
        print(
            f"Model      : {settings.RAG_EMBEDDING_MODEL}"
        )
        print(
            f"Dimensions : {settings.RAG_EMBEDDING_DIMENSIONS}"
        )

        total_active = await collection.count_documents({"status": "active"})

        matching_count = await collection.count_documents({
            "status": "active",
            "embedding_model": settings.RAG_EMBEDDING_MODEL,
            "embedding_dimensions": settings.RAG_EMBEDDING_DIMENSIONS,
        })

        mismatched_count = await collection.count_documents({
            "status": "active",
            "$or": [
                {"embedding_model": {"$ne": settings.RAG_EMBEDDING_MODEL}},
                {"embedding_dimensions": {"$ne": settings.RAG_EMBEDDING_DIMENSIONS}},
                {"embedding_model": {"$exists": False}},
                {"embedding_dimensions": {"$exists": False}},
            ],
        })

        print("\nMigration status:")
        print(f"Total active chunks               : {total_active}")
        print(f"Chunks matching current config   : {matching_count}")
        print(f"Chunks requiring migration       : {mismatched_count}")
        if mismatched_count == 0:
            print("\n[OK] All active chunks are fully migrated to current embedding configuration!")
        else:
            print(f"\n[WARNING] {mismatched_count} chunks require migration to {settings.RAG_EMBEDDING_MODEL} ({settings.RAG_EMBEDDING_DIMENSIONS} dims).")

    finally:
        await disconnect_db()




if __name__ == "__main__":
    asyncio.run(main())