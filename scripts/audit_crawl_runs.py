import asyncio

from app.core.database import connect_db, disconnect_db, get_database


async def main():
    await connect_db()

    try:
        db = get_database()

        runs = await (
            db.crawl_runs
            .find({})
            .sort("started_at", -1)
            .limit(10)
            .to_list(length=10)
        )

        print("=" * 70)
        print("CRAWL RUN AUDIT")
        print("=" * 70)

        if not runs:
            print("No crawl runs found.")
            return

        for index, run in enumerate(runs, start=1):
            print(f"\nRun #{index}")
            print("-" * 70)

            print(f"ID                  : {run.get('_id')}")
            print(f"Status              : {run.get('status')}")
            print(f"Started             : {run.get('started_at')}")
            print(f"Finished            : {run.get('finished_at')}")
            print(f"URLs discovered     : {run.get('urls_discovered', 0)}")
            print(f"URLs crawled        : {run.get('urls_crawled', 0)}")
            print(f"Documents changed   : {run.get('documents_changed', 0)}")
            print(f"Documents skipped   : {run.get('documents_skipped', 0)}")
            print(f"Chunks created      : {run.get('chunks_created', 0)}")

            errors = run.get("errors") or []
            print(f"Errors              : {len(errors)}")

            if errors:
                for error in errors[:5]:
                    print(
                        f"  - {error.get('url')}: "
                        f"{error.get('error')}"
                    )

    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())