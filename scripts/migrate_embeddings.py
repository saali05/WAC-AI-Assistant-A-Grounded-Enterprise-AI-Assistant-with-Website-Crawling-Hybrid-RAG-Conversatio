import argparse
import asyncio
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import connect_db, disconnect_db
from app.core.logging import logger
from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.indexing.indexer import DocumentIndexer
from app.services.crawl_service import CrawlService


async def run_migration(
    batch_size: int = 25,
    force_all: bool = False,
    dry_run: bool = False,
    delay: float = 0.0,
    provider: Optional[str] = None,
    confirm: bool = False,
):
    await connect_db()

    try:
        chosen_provider = (provider or getattr(settings, "RAG_EMBEDDING_PROVIDER", "gemini")).strip().lower()

        if chosen_provider == "local":
            target_model = getattr(settings, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            target_dimensions = getattr(settings, "LOCAL_EMBEDDING_DIMENSIONS", 768)
        else:
            target_model = settings.RAG_EMBEDDING_MODEL
            target_dimensions = settings.RAG_EMBEDDING_DIMENSIONS

        embedding_service = EmbeddingService(
            provider=chosen_provider,
            model=target_model,
            dimensions=target_dimensions,
        )
        indexer = DocumentIndexer(embedding_service=embedding_service)
        crawl_service = CrawlService(indexer=indexer)

        print("=" * 70)
        print("WAC RAG EMBEDDING MODEL MIGRATION")
        print("=" * 70)
        print(f"Target Embedding Provider   : {chosen_provider}")
        print(f"Target Embedding Model      : {target_model}")
        print(f"Target Embedding Dimensions : {target_dimensions}")
        print(f"Batch Size                  : {batch_size}")
        print(f"Batch Delay                 : {delay}s")
        print(f"Force All Chunks            : {force_all}")
        print(f"Dry Run Mode                : {dry_run}")
        print("-" * 70)

        print("\n" + "!" * 70)
        print("[CRITICAL WARNING] VECTOR SPACE INCOMPATIBILITY")
        print("Changing embedding providers transforms the mathematical vector space.")
        print("Gemini vectors and Local BGE vectors cannot be mixed in similarity searches.")
        print("If migrating to a new provider, ALL active chunks must be re-indexed before querying.")
        print("!" * 70 + "\n")

        total_active = await crawl_service.chunk_repo.get_active_chunks_count()
        mismatched_count = await crawl_service.chunk_repo.get_mismatched_chunks_count(
            model=target_model,
            dimensions=target_dimensions,
        )

        print(f"Total Active Chunks         : {total_active}")
        print(f"Chunks Requiring Migration  : {mismatched_count}")
        print("-" * 70)

        if mismatched_count == 0 and not force_all:
            print("[OK] All active chunks are already using the configured embedding model and dimensions.")
            print("No migration needed.")
            return

        if dry_run:
            print("[INFO] DRY RUN COMPLETE: No database modifications were made.")
            return

        if not confirm:
            print("[SAFETY ABORT] Execution requires the '--confirm' flag to modify MongoDB embeddings.")
            print("Run with '--dry-run' to inspect or add '--confirm' to perform live re-indexing.")
            return

        print(f"[START] Starting migration for {mismatched_count if not force_all else total_active} chunks...")
        result = await crawl_service.reindex_all(
            batch_size=batch_size,
            force_all=force_all,
            model=target_model,
            dimensions=target_dimensions,
            delay=delay,
        )

        print("\n" + "=" * 70)
        print("MIGRATION SUMMARY")
        print("=" * 70)
        print(f"Chunks Found     : {result.get('chunks_found', 0)}")
        print(f"Chunks Processed : {result.get('chunks_processed', 0)}")
        print(f"Chunks Failed    : {result.get('chunks_failed', 0)}")
        print(f"Chunks Remaining : {result.get('chunks_remaining', 0)}")
        print("=" * 70)

        if result.get("chunks_remaining", 0) == 0 and result.get("chunks_failed", 0) == 0:
            print("[SUCCESS] Migration successfully completed!")
        else:
            print("[WARNING] Migration finished with remaining or failed chunks. Run again to resume.")

    finally:
        await disconnect_db()


def main():
    parser = argparse.ArgumentParser(description="WAC RAG Embedding Migration Script")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "local"],
        default=None,
        help="Target embedding provider (gemini or local, default: settings.RAG_EMBEDDING_PROVIDER)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for embedding generation (default: 25)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between embedding batches for rate-limit smoothing (default: 0.0)",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force re-indexing of all chunks regardless of current embedding model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect mismatched chunks without generating embeddings or updating database",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation required to write updated embeddings to MongoDB",
    )

    args = parser.parse_args()

    asyncio.run(
        run_migration(
            batch_size=args.batch_size,
            force_all=args.force_all,
            dry_run=args.dry_run,
            delay=args.delay,
            provider=args.provider,
            confirm=args.confirm,
        )
    )


if __name__ == "__main__":
    main()
