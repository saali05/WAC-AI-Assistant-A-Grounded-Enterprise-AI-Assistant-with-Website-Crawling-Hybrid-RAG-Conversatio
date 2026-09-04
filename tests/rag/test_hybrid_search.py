import pytest
from app.rag.models import RetrievedChunk
from app.rag.retrieval.context_builder import ContextBuilder
from app.rag.retrieval.query_rewriter import QueryRewriter
from app.rag.retrieval.reranker import FusionReranker


def test_query_rewriter():
    original = "What about ecommerce?"
    history = (
        "User: What services does WAC provide?\n"
        "Assistant: WAC provides custom AI and software services."
    )

    rewritten = QueryRewriter.rewrite(
        original,
        history,
    )

    assert rewritten == "What about ecommerce?"


def test_context_builder():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="WAC develops custom AI solutions.",
            title="WAC Services",
            heading_path=["Services", "AI"],
            url="https://webandcrafts.com/services",
            canonical_url="https://webandcrafts.com/services",
            score=0.91
        )
    ]
    context, sources = ContextBuilder.build_context_and_sources(chunks)
    assert "WAC KNOWLEDGE CONTEXT" in context
    assert "https://webandcrafts.com/services" in context
    assert len(sources) == 1
    assert sources[0].title == "WAC Services"


@pytest.mark.anyio
async def test_fusion_reranker():
    reranker = FusionReranker()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="General content",
            title="WAC Services",
            heading_path=["AI"],
            url="https://webandcrafts.com/services",
            canonical_url="https://webandcrafts.com/services",
            score=0.8
        )
    ]
    reranked = await reranker.rerank("AI Services", chunks)
    assert len(reranked) == 1
    assert reranked[0].score >= 0.8
