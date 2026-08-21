from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Query,
)

from app.schemas.rag import (
    CrawlRequest,
    CrawlStatusResponse,
    DocumentListResponse,
    DocumentListItem,
)

from app.services.crawl_service import CrawlService

from app.repositories.rag_repository import (
    RAGDocumentRepository,
    RAGChunkRepository,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG Admin"],
)


@router.post(
    "/crawl",
    response_model=dict,
)
async def trigger_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger website crawling and indexing
    in the background.
    """

    crawl_service = CrawlService()

    try:

        background_tasks.add_task(
            crawl_service.start_crawl,
            request.start_urls,
        )

        return {
            "message": (
                "Website crawl started "
                "successfully in background."
            ),
            "status": "running",
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to start crawl: "
                f"{exc}"
            ),
        )


@router.get(
    "/crawl/status",
    response_model=CrawlStatusResponse,
)
async def get_crawl_status(
    run_id: str | None = Query(
        None,
        description="Optional Crawl Run ID",
    ),
):
    """
    Retrieve current or latest crawl status.
    """

    crawl_service = CrawlService()

    try:

        status_data = (
            await crawl_service.get_status(
                run_id
            )
        )

        if not status_data:

            return CrawlStatusResponse(
                status="idle"
            )

        return CrawlStatusResponse(
            run_id=status_data.get("id"),
            status=status_data.get(
                "status",
                "completed",
            ),
            started_at=status_data.get(
                "started_at"
            ),
            finished_at=status_data.get(
                "finished_at"
            ),
            urls_discovered=status_data.get(
                "urls_discovered",
                0,
            ),
            urls_crawled=status_data.get(
                "urls_crawled",
                0,
            ),
            documents_changed=status_data.get(
                "documents_changed",
                0,
            ),
            documents_skipped=status_data.get(
                "documents_skipped",
                0,
            ),
            chunks_created=status_data.get(
                "chunks_created",
                0,
            ),
            errors=status_data.get(
                "errors",
                [],
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post(
    "/reindex",
)
async def reindex_documents():
    """
    Re-generate embeddings for all active
    chunks using the currently configured
    embedding model.

    This does NOT crawl the website.
    """

    crawl_service = CrawlService()

    try:

        result = (
            await crawl_service.reindex_all()
        )

        return {
            "message": (
                "RAG embeddings re-indexed "
                "successfully."
            ),
            **result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG embedding reindex failed: "
                f"{exc}"
            ),
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    skip: int = Query(
        0,
        ge=0,
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
    ),
    status: str | None = Query(
        "active"
    ),
):
    """
    List indexed RAG documents.
    """

    doc_repo = RAGDocumentRepository()

    try:

        docs = await doc_repo.get_all(
            skip=skip,
            limit=limit,
            status=status,
        )

        total = await doc_repo.count(
            status=status
        )

        items = [
            DocumentListItem(
                id=document["id"],
                url=document.get(
                    "url",
                    "",
                ),
                canonical_url=document.get(
                    "canonical_url",
                    "",
                ),
                title=document.get(
                    "title",
                    "",
                ),
                domain=document.get(
                    "domain",
                    "",
                ),
                status=document.get(
                    "status",
                    "active",
                ),
                word_count=document.get(
                    "word_count",
                    0,
                ),
                last_crawled_at=document.get(
                    "last_crawled_at"
                ),
            )
            for document in docs
        ]

        return DocumentListResponse(
            documents=items,
            total=total,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.get(
    "/documents/{doc_id}",
)
async def get_document_details(
    doc_id: str,
):
    """
    Get details of an indexed document.
    """

    doc_repo = RAGDocumentRepository()

    document = await doc_repo.get_by_id(
        doc_id
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document


@router.delete(
    "/documents/{doc_id}",
)
async def delete_document(
    doc_id: str,
):
    """
    Delete a document and its associated
    chunks.
    """

    doc_repo = RAGDocumentRepository()
    chunk_repo = RAGChunkRepository()

    deleted_document = (
        await doc_repo.delete(doc_id)
    )

    if not deleted_document:

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    deleted_chunks = (
        await chunk_repo
        .delete_by_document_id(
            doc_id
        )
    )

    return {
        "message": (
            "Successfully deleted document "
            "and associated chunks."
        ),
        "document_id": doc_id,
        "chunks_deleted": deleted_chunks,
    }