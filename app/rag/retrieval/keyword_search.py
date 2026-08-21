from typing import Optional
from app.core.config import settings
from app.repositories.rag_repository import RAGChunkRepository


class KeywordSearch:
    """Full-text / lexical keyword search over rag_chunks content and title."""

    def __init__(self, chunk_repo: Optional[RAGChunkRepository] = None) -> None:
        self.chunk_repo = chunk_repo or RAGChunkRepository()

    async def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Perform keyword search over active chunks."""
        k = top_k or settings.RAG_TOP_K_KEYWORD
        if not query or not query.strip():
            return []

        results = await self.chunk_repo.keyword_search(query_str=query, top_k=k)
        return results
