import asyncio
import pytest
from app.core.database import connect_db, disconnect_db
from app.rag.validation.relevance import WACRelevanceGate
from app.services.rag_service import RAGService
from app.ai.service import AIService
from app.rag.models import RetrievedChunk
from datetime import datetime, timezone, timedelta


async def test_relevance_gate():
    """Test WAC domain relevance classification rules."""
    
    # TEST 1: WAC services question -> True
    is_wac, refusal = WACRelevanceGate.evaluate("What services does WAC provide?")
    assert is_wac is True
    assert refusal is None

    # TEST 2: Digital marketing offer -> True
    is_wac, refusal = WACRelevanceGate.evaluate("Does WAC offer digital marketing?")
    assert is_wac is True

    # TEST 3: WAC technologies -> True
    is_wac, refusal = WACRelevanceGate.evaluate("What technologies does WAC work with?")
    assert is_wac is True

    # TEST 4: Capital of France -> False
    is_wac, refusal = WACRelevanceGate.evaluate("What is the capital of France?")
    assert is_wac is False
    assert refusal is not None

    # TEST 5: What is React? -> False (generic definition without WAC context)
    is_wac, refusal = WACRelevanceGate.evaluate("What is React?")
    assert is_wac is False
    assert refusal is not None

    # TEST 6: What React services does WAC provide? -> True
    is_wac, refusal = WACRelevanceGate.evaluate("What React services does WAC provide?")
    assert is_wac is True

    # TEST 7: Who are WAC's clients? -> True
    is_wac, refusal = WACRelevanceGate.evaluate("Who are WAC's clients?")
    assert is_wac is True

    # TEST 8: Tell me something about WAC not in knowledge base -> True (WAC question)
    is_wac, refusal = WACRelevanceGate.evaluate("Tell me something about WAC that isn't in your knowledge base.")
    assert is_wac is True

    # TEST 9: Affirmative follow-up with history -> True
    history = "User: What UI/UX solutions does Webandcrafts offer?\nAssistant: We design enterprise UI/UX. Would you like to discuss specific industry solutions?"
    is_wac, refusal = WACRelevanceGate.evaluate("yes i would like to discuss", conversation_history=history)
    assert is_wac is True
    assert refusal is None

    # TEST 10: Tell me more with history -> True
    is_wac, refusal = WACRelevanceGate.evaluate("tell me more", conversation_history=history)
    assert is_wac is True
    assert refusal is None


async def test_freshness_sorting():
    """TEST 10: Reranker tie-breaking prefers newer valid source content."""
    from app.rag.retrieval.reranker import FusionReranker

    reranker = FusionReranker()
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=30)

    chunk_old = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        content="Old content",
        title="WAC Services",
        heading_path=[],
        url="https://webandcrafts.com/services",
        canonical_url="https://webandcrafts.com/services",
        score=0.8,
        vector_score=0.8,
        keyword_score=0.5,
        fusion_score=0.01,
        updated_at=older
    )

    chunk_new = RetrievedChunk(
        chunk_id="c2",
        document_id="d2",
        content="New content",
        title="WAC Services",
        heading_path=[],
        url="https://webandcrafts.com/services",
        canonical_url="https://webandcrafts.com/services",
        score=0.8,
        vector_score=0.8,
        keyword_score=0.5,
        fusion_score=0.01,
        updated_at=now
    )

    reranked = await reranker.rerank(
        query="WAC Services",
        chunks=[chunk_old, chunk_new],
        top_k=2
    )

    assert len(reranked) == 2
    # The newer chunk should be first when scores are tied
    assert reranked[0].chunk_id == "c2"


async def main():
    await connect_db()
    try:
        print("Running test_relevance_gate...")
        await test_relevance_gate()
        print("PASSED: test_relevance_gate")

        print("Running test_freshness_sorting...")
        await test_freshness_sorting()
        print("PASSED: test_freshness_sorting")

        ai_service = AIService()

        # TEST 4 (Integration): Non-WAC query returns immediate refusal
        print("\nTesting Non-WAC query ('What is the capital of France?')...")
        response, rag_result = await ai_service.chat("What is the capital of France?")
        print(f"Is Relevant: {rag_result.is_relevant}, Has Context: {rag_result.has_context}")
        print(f"Refusal/Response: {response.content}")
        assert rag_result.is_relevant is False
        assert "WAC AI Assistant" in response.content

        # TEST 5 (Integration): Generic tech definition returns refusal
        print("\nTesting Generic Tech query ('What is React?')...")
        response, rag_result = await ai_service.chat("What is React?")
        print(f"Is Relevant: {rag_result.is_relevant}, Has Context: {rag_result.has_context}")
        assert rag_result.is_relevant is False

        # TEST 9 (Integration): WAC question with non-existent topic returns refusal
        print("\nTesting Weak Retrieval query ('What is WAC rocket science aerospace service?')...")
        response, rag_result = await ai_service.chat("What is WAC rocket science aerospace service?")
        print(f"Is Relevant: {rag_result.is_relevant}, Has Context: {rag_result.has_context}")
        assert rag_result.has_context is False

    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())
