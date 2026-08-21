from datetime import datetime, UTC
from typing import Optional

from app.core.config import settings
from app.core.logging import logger

from app.rag.crawler.crawler import WebCrawler
from app.rag.extraction.metadata_extractor import MetadataExtractor
from app.rag.indexing.indexer import DocumentIndexer
from app.rag.models import (
    CrawlRunModel,
    CrawlRunError,
)

from app.repositories.rag_repository import (
    CrawlRunRepository,
    RAGDocumentRepository,
    RAGChunkRepository,
)


class CrawlService:
    """
    High-level service managing:

    - Website crawling
    - Document extraction
    - Document indexing
    - Embedding generation
    - Embedding re-indexing
    """

    def __init__(
        self,
        crawler: Optional[WebCrawler] = None,
        indexer: Optional[DocumentIndexer] = None,
        crawl_repo: Optional[CrawlRunRepository] = None,
        doc_repo: Optional[RAGDocumentRepository] = None,
        chunk_repo: Optional[RAGChunkRepository] = None,
    ) -> None:

        self.crawler = (
            crawler
            or WebCrawler()
        )

        self.indexer = (
            indexer
            or DocumentIndexer(
                doc_repo=doc_repo,
                chunk_repo=chunk_repo,
            )
        )

        self.crawl_repo = (
            crawl_repo
            or CrawlRunRepository()
        )

        self.doc_repo = (
            doc_repo
            or RAGDocumentRepository()
        )

        self.chunk_repo = (
            chunk_repo
            or RAGChunkRepository()
        )

    # ==========================================================
    # WEBSITE CRAWLING
    # ==========================================================

    async def start_crawl(
        self,
        start_urls: Optional[list[str]] = None,
    ) -> str:

        """
        Execute complete website crawl and
        indexing workflow asynchronously.
        """

        urls = (
            start_urls
            or [
                f"https://{domain}"
                for domain in settings.allowed_domains_list
            ]
        )

        run_model = CrawlRunModel(
            status="running",
            started_at=datetime.now(UTC),
        )

        run_id = await self.crawl_repo.create(
            run_model
        )

        logger.info(
            "🚀 Starting website crawl run "
            f"[{run_id}] for target domains: "
            f"{settings.RAG_ALLOWED_DOMAINS}"
        )

        crawled_count = 0
        changed_count = 0
        skipped_count = 0
        total_chunks = 0

        errors: list[CrawlRunError] = []

        try:

            pages = await self.crawler.crawl(
                urls
            )

            discovered_count = (
                len(pages)
                + len(self.crawler.visited_urls)
            )

            for page in pages:

                crawled_count += 1

                try:

                    doc_model, extracted = (
                        MetadataExtractor
                        .extract_document_model(
                            raw_html=page.html,
                            url=page.url,
                            http_status=page.http_status,
                        )
                    )

                    if doc_model.status != "active":

                        skipped_count += 1

                        continue

                    status, doc_id, num_chunks = (
                        await self.indexer.index_document(
                            doc_model,
                            extracted,
                        )
                    )

                    if status == "indexed":

                        changed_count += 1
                        total_chunks += num_chunks

                    else:

                        skipped_count += 1

                except Exception as exc:

                    logger.error(
                        "Error indexing page "
                        f"{page.url}: {exc}"
                    )

                    errors.append(
                        CrawlRunError(
                            url=page.url,
                            error=str(exc),
                        )
                    )

            # --------------------------------------------------
            # CRAWLER FAILURES
            # --------------------------------------------------

            for (
                fail_url,
                fail_err,
            ) in self.crawler.failed_urls.items():

                errors.append(
                    CrawlRunError(
                        url=fail_url,
                        error=fail_err,
                    )
                )

            # --------------------------------------------------
            # COMPLETE CRAWL RUN
            # --------------------------------------------------

            update_data = {
                "finished_at": datetime.now(UTC),
                "status": "completed",
                "urls_discovered": discovered_count,
                "urls_crawled": crawled_count,
                "documents_changed": changed_count,
                "documents_skipped": skipped_count,
                "chunks_created": total_chunks,
                "errors": [
                    error.model_dump()
                    for error in errors
                ],
            }

            await self.crawl_repo.update(
                run_id,
                update_data,
            )

            logger.info(
                "✅ Crawl run "
                f"[{run_id}] completed: "
                f"{crawled_count} pages crawled, "
                f"{changed_count} updated, "
                f"{total_chunks} chunks indexed."
            )

            return run_id

        except Exception as exc:

            logger.exception(
                f"❌ Crawl run [{run_id}] failed: "
                f"{exc}"
            )

            await self.crawl_repo.update(
                run_id,
                {
                    "finished_at": datetime.now(
                        UTC
                    ),
                    "status": "failed",
                    "urls_crawled": crawled_count,
                    "documents_changed": changed_count,
                    "documents_skipped": skipped_count,
                    "chunks_created": total_chunks,
                    "errors": [
                        error.model_dump()
                        for error in errors
                    ]
                    + [
                        {
                            "url": "",
                            "error": str(exc),
                        }
                    ],
                },
            )

            raise

    # ==========================================================
    # EMBEDDING REINDEX
    # ==========================================================

    async def reindex_all(
        self,
        batch_size: int = 25,
    ) -> dict[str, int]:
        """
        Re-generate embeddings for all active chunks.

        IMPORTANT:

        This does NOT crawl the website.

        Existing chunk.content is reused.

        This is specifically intended for:
        - embedding model migration
        - embedding dimension migration
        - repairing invalid/fallback embeddings
        """

        logger.info(
            "🔄 Starting RAG embedding reindex..."
        )

        total_chunks = (
            await self.chunk_repo
            .get_active_chunks_count()
        )

        if total_chunks == 0:

            logger.warning(
                "No active RAG chunks found."
            )

            return {
                "chunks_found": 0,
                "chunks_processed": 0,
                "chunks_failed": 0,
            }

        logger.info(
            "Found "
            f"{total_chunks} active chunks "
            "requiring embedding generation."
        )

        processed = 0
        failed = 0

        cursor = (
            await self.chunk_repo
            .get_active_chunks(
                batch_size=batch_size
            )
        )

        while True:

            chunks = await cursor.to_list(
                length=batch_size
            )

            if not chunks:
                break

            valid_chunks = []

            for chunk in chunks:

                content = (
                    chunk.get("content")
                    or ""
                ).strip()

                if not content:

                    logger.warning(
                        "Skipping chunk with empty "
                        f"content: {chunk.get('_id')}"
                    )

                    failed += 1

                    continue

                valid_chunks.append(
                    chunk
                )

            if not valid_chunks:
                continue

            texts = [
                chunk["content"].strip()
                for chunk in valid_chunks
            ]

            try:

                embeddings = (
                    await self.indexer
                    .embedding_service
                    .get_batch_embeddings(
                        texts
                    )
                )

                if len(embeddings) != len(
                    valid_chunks
                ):

                    raise RuntimeError(
                        "Embedding count mismatch: "
                        f"expected "
                        f"{len(valid_chunks)}, "
                        f"received "
                        f"{len(embeddings)}"
                    )

                for chunk, embedding in zip(
                    valid_chunks,
                    embeddings,
                ):

                    chunk_id = str(
                        chunk["_id"]
                    )

                    expected_dimensions = (
                        settings
                        .RAG_EMBEDDING_DIMENSIONS
                    )

                    actual_dimensions = len(
                        embedding
                    )

                    if (
                        actual_dimensions
                        != expected_dimensions
                    ):

                        raise RuntimeError(
                            "Embedding dimension "
                            "mismatch for chunk "
                            f"{chunk_id}: "
                            f"expected="
                            f"{expected_dimensions}, "
                            f"actual="
                            f"{actual_dimensions}"
                        )

                    updated = (
                        await self.chunk_repo
                        .update_embedding(
                            chunk_id=chunk_id,
                            embedding=embedding,
                            model=(
                                settings
                                .RAG_EMBEDDING_MODEL
                            ),
                            dimensions=(
                                expected_dimensions
                            ),
                        )
                    )

                    if updated:

                        processed += 1

                    else:

                        failed += 1

            except Exception as exc:

                failed += len(
                    valid_chunks
                )

                logger.error(
                    "❌ Failed to reindex "
                    f"batch of {len(valid_chunks)} "
                    f"chunks: {exc}"
                )

                # Don't continue silently if Gemini
                # itself has failed. This prevents
                # thousands of failed API requests.
                raise

            logger.info(
                "🔄 Reindex progress: "
                f"{processed + failed}/"
                f"{total_chunks} "
                f"| successful={processed} "
                f"| failed={failed}"
            )

        logger.info(
            "✅ RAG embedding reindex completed | "
            f"found={total_chunks} | "
            f"processed={processed} | "
            f"failed={failed}"
        )

        return {
            "chunks_found": total_chunks,
            "chunks_processed": processed,
            "chunks_failed": failed,
        }

    # ==========================================================
    # CRAWL STATUS
    # ==========================================================

    async def get_status(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[dict]:

        """
        Get a specific crawl run or the latest crawl run.
        """

        if run_id:

            return await self.crawl_repo.get_by_id(
                run_id
            )

        return await self.crawl_repo.get_latest()