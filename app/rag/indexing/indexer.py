from datetime import datetime, UTC
from typing import Optional
from app.core.logging import logger
from app.rag.chunking.semantic_chunker import SemanticChunker
# from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.extraction.html_extractor import ExtractedHTML
from app.rag.models import RAGDocumentModel, RAGChunkModel
from app.repositories.rag_repository import RAGDocumentRepository, RAGChunkRepository


class DocumentIndexer:
    """Document versioning and vector chunk indexer."""

    def __init__(
        self,
        doc_repo: Optional[RAGDocumentRepository] = None,
        chunk_repo: Optional[RAGChunkRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[SemanticChunker] = None
    ) -> None:
        self.doc_repo = doc_repo or RAGDocumentRepository()
        self.chunk_repo = chunk_repo or RAGChunkRepository()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or SemanticChunker()

    async def index_document(
        self,
        doc_model: RAGDocumentModel,
        extracted: ExtractedHTML
    ) -> tuple[str, str, int]:
        """
        Index document:
        - Returns (status, document_id, chunks_created) where status is "skipped" or "indexed".
        """
        existing_doc = await self.doc_repo.get_by_canonical_url(doc_model.canonical_url)

        if existing_doc:
            doc_id = existing_doc["id"]
            old_hash = existing_doc.get("content_hash")

            if old_hash == doc_model.content_hash:
                # Document is unchanged: skip re-embedding
                await self.doc_repo.update(doc_id, {"last_crawled_at": datetime.now(UTC)})
                logger.info(f"Skipped unchanged document: {doc_model.canonical_url}")
                return "skipped", doc_id, 0

            # Document has changed: create new version
            new_version = existing_doc.get("version", 1) + 1
            doc_model.version = new_version
            doc_model.last_changed_at = datetime.now(UTC)

            # Deactivate previous active chunks
            await self.chunk_repo.deactivate_by_document_id(doc_id)

            # Chunk document content
            chunks = self.chunker.chunk_document(
                document_id=doc_id,
                extracted=extracted,
                url=doc_model.url,
                canonical_url=doc_model.canonical_url,
                version=new_version
            )

            # Embed chunks
            if chunks:
                chunk_texts = [c.content for c in chunks]
                embeddings = await self.embedding_service.get_batch_embeddings(chunk_texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb

                await self.chunk_repo.create_many(chunks)

            # Update document record
            update_data = doc_model.model_dump(exclude={"id"})
            await self.doc_repo.update(doc_id, update_data)

            logger.info(f"Re-indexed changed document {doc_model.canonical_url} (version {new_version}, {len(chunks)} chunks)")
            return "indexed", doc_id, len(chunks)

        else:
            # Brand new document
            doc_id = await self.doc_repo.create(doc_model)

            chunks = self.chunker.chunk_document(
                document_id=doc_id,
                extracted=extracted,
                url=doc_model.url,
                canonical_url=doc_model.canonical_url,
                version=1
            )

            if chunks:
                chunk_texts = [c.content for c in chunks]
                embeddings = await self.embedding_service.get_batch_embeddings(chunk_texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb

                await self.chunk_repo.create_many(chunks)

            logger.info(f"Indexed new document {doc_model.canonical_url} ({len(chunks)} chunks)")
            return "indexed", doc_id, len(chunks)
