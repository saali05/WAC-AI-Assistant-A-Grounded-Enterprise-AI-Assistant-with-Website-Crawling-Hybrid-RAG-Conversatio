from typing import Optional
from app.core.config import settings
from app.rag.embeddings.embedding_service import EmbeddingService
from app.repositories.rag_repository import RAGChunkRepository


class VectorSearch:
    """Vector search over rag_chunks embedding index."""

    def __init__(
        self,
        chunk_repo: Optional[RAGChunkRepository] = None,
        embedding_service: Optional[EmbeddingService] = None
    ) -> None:
        self.chunk_repo = chunk_repo or RAGChunkRepository()
        self.embedding_service = embedding_service or EmbeddingService()

    async def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Generate query vector and search rag_chunks."""
        k = top_k or settings.RAG_TOP_K_VECTOR
        if not query or not query.strip():
            return []

        query_vector = await self.embedding_service.get_embedding(query)
        results = await self.chunk_repo.vector_search(query_vector=query_vector, top_k=k)
        return results
