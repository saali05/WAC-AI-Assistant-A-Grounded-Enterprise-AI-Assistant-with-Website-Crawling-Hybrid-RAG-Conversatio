from typing import Optional
from app.core.config import settings
from app.rag.models import RetrievedChunk
from app.rag.retrieval.keyword_search import KeywordSearch
from app.rag.retrieval.vector_search import VectorSearch


class HybridSearch:
    """Hybrid Search using Reciprocal Rank Fusion (RRF) combining vector search and keyword search."""

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        keyword_search: Optional[KeywordSearch] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        rrf_k: int = 60
    ) -> None:
        self.vector_search = vector_search or VectorSearch()
        self.keyword_search = keyword_search or KeywordSearch()
        self.vector_weight = vector_weight if vector_weight is not None else settings.RAG_VECTOR_WEIGHT
        self.keyword_weight = keyword_weight if keyword_weight is not None else settings.RAG_KEYWORD_WEIGHT
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        """Perform hybrid search and fuse ranks using RRF."""
        k = top_k or settings.RAG_TOP_K_VECTOR

        # Run vector and keyword searches
        vector_results = await self.vector_search.search(query, top_k=k)
        keyword_results = await self.keyword_search.search(query, top_k=k)

        chunk_dict: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}
        vec_scores: dict[str, float] = {}
        kw_scores: dict[str, float] = {}

        # Process vector ranks
        for rank, item in enumerate(vector_results):
            cid = item["id"]
            chunk_dict[cid] = item
            score = item.get("score", 0.0)
            vec_scores[cid] = score
            rrf_score = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

        # Process keyword ranks
        for rank, item in enumerate(keyword_results):
            cid = item["id"]
            if cid not in chunk_dict:
                chunk_dict[cid] = item
            score = item.get("score", 0.0)
            kw_scores[cid] = score
            rrf_score = self.keyword_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

        # Convert to RetrievedChunk objects
        fused_chunks: list[RetrievedChunk] = []
        for cid, item in chunk_dict.items():
            fusion_score = rrf_scores.get(cid, 0.0)
            v_score = vec_scores.get(cid, 0.0)
            k_score = kw_scores.get(cid, 0.0)

            # Combined normalized score
            final_score = max(v_score, k_score, fusion_score * 50)

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=item.get("document_id", ""),
                    content=item.get("content", ""),
                    title=item.get("title", ""),
                    heading_path=item.get("heading_path", []),
                    url=item.get("url", ""),
                    canonical_url=item.get("canonical_url", ""),
                    score=final_score,
                    vector_score=v_score,
                    keyword_score=k_score,
                    fusion_score=fusion_score
                )
            )

        fused_chunks.sort(key=lambda x: x.score, reverse=True)
        return fused_chunks[:k]
