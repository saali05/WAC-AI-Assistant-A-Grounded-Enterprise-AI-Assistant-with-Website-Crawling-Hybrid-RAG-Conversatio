import pytest
from unittest.mock import AsyncMock, patch

from app.ai.schemas import AIResponse, AIUsage
from app.ai.service import AIService
from app.rag.models import RAGResult, RetrievedChunk
from app.rag.retrieval.hybrid_search import HybridSearch
from app.rag.retrieval.query_rewriter import QueryRewriter
from app.rag.retrieval.reranker import FusionReranker
from app.rag.validation.relevance import WACRelevanceGate
from app.repositories.rag_repository import RAGChunkRepository
from app.services.rag_service import RAGService


# ==============================================================================
# 24 COMPREHENSIVE TESTS FOR RETRIEVAL ARCHITECTURE
# ==============================================================================

def test_01_what_technologies_does_wac_use():
    """1. 'What technologies does WAC use?' -> Tech detection & retrieval expansion."""
    query = "What technologies does WAC use?"
    assert QueryRewriter.is_technology_query(query) is True
    assert QueryRewriter.rewrite(query) == query
    expanded = QueryRewriter.expand_for_retrieval(query)
    assert "React" in expanded and "Node.js" in expanded


def test_02_what_technology_does_wac_use():
    """2. 'What technology does WAC use?' (singular) -> Tech detection & expansion."""
    query = "What technology does WAC use?"
    assert QueryRewriter.is_technology_query(query) is True
    assert QueryRewriter.rewrite(query) == query
    expanded = QueryRewriter.expand_for_retrieval(query)
    assert "React" in expanded and "Laravel" in expanded


def test_03_which_technologies_does_wac_use():
    """3. 'Which technologies does WAC use?' -> Tech detection & expansion."""
    query = "Which technologies does WAC use?"
    assert QueryRewriter.is_technology_query(query) is True
    expanded = QueryRewriter.expand_for_retrieval(query)
    assert "Angular" in expanded and "AWS" in expanded


def test_04_what_is_wac_technology_stack():
    """4. 'What is WAC's technology stack?' -> Tech stack detection & expansion."""
    query = "What is WAC's technology stack?"
    assert QueryRewriter.is_technology_query(query) is True
    expanded = QueryRewriter.expand_for_retrieval(query)
    assert "Flutter" in expanded and "MongoDB" in expanded


def test_05_does_wac_use_react():
    """5. 'Does WAC use React?' -> Specific technology query."""
    query = "Does WAC use React?"
    is_wac, refusal = WACRelevanceGate.evaluate(query)
    assert is_wac is True
    terms = RAGChunkRepository._extract_keyword_terms(query)
    assert "React" in terms


def test_06_does_wac_use_nodejs():
    """6. 'Does WAC use Node.js?' -> Punctuation-preserved technology query."""
    query = "Does WAC use Node.js?"
    is_wac, refusal = WACRelevanceGate.evaluate(query)
    assert is_wac is True
    terms = RAGChunkRepository._extract_keyword_terms(query)
    assert "Node.js" in terms


def test_07_does_wac_use_laravel():
    """7. 'Does WAC use Laravel?' -> Laravel specific technology query."""
    query = "Does WAC use Laravel?"
    is_wac, _ = WACRelevanceGate.evaluate(query)
    assert is_wac is True
    terms = RAGChunkRepository._extract_keyword_terms(query)
    assert "Laravel" in terms


def test_08_ecommerce_technology():
    """8. 'Tell me about WAC's ecommerce technology.' -> Domain & ecommerce tech query."""
    query = "Tell me about WAC's ecommerce technology."
    is_wac, _ = WACRelevanceGate.evaluate(query)
    assert is_wac is True
    terms = RAGChunkRepository._extract_keyword_terms(query)
    assert "ecommerce" in [t.lower() for t in terms]
    assert "technology" in [t.lower() for t in terms]


def test_09_what_services_does_wac_provide():
    """9. 'What services does WAC provide?' -> General services query."""
    query = "What services does WAC provide?"
    is_wac, _ = WACRelevanceGate.evaluate(query)
    assert is_wac is True
    assert QueryRewriter.rewrite(query) == query


def test_10_what_about_digital_marketing():
    """10. 'What about digital marketing?' -> Explicit new topic follow-up."""
    history = "User: What services does WAC provide?\nAssistant: WAC provides custom AI and software."
    assert QueryRewriter.rewrite("What about digital marketing?", history) == "What about digital marketing?"


def test_11_tell_me_more():
    """11. 'tell me more' -> Pure continuation resolving to previous topic."""
    history = "User: What services does WAC provide?\nAssistant: WAC provides web development."
    assert QueryRewriter.rewrite("tell me more", history) == "What services does WAC provide?"


def test_12_what_about_it():
    """12. 'what about it?' -> Pronoun contextual follow-up."""
    history = "User: What services does WAC provide?\nAssistant: WAC provides cloud computing."
    assert QueryRewriter.rewrite("what about it?", history) == "What services does WAC provide? what about it?"


def test_13_unrelated_question():
    """13. Unrelated question -> Relevance gate refusal."""
    is_wac, refusal = WACRelevanceGate.evaluate("What is the recipe for chocolate cake?")
    assert is_wac is False
    assert refusal is not None


def test_14_out_of_domain_question():
    """14. Out-of-domain question ('What is the capital of France?') -> Gating refusal."""
    is_wac, refusal = WACRelevanceGate.evaluate("What is the capital of France?")
    assert is_wac is False
    assert refusal is not None


@pytest.mark.anyio
async def test_15_keyword_only_candidate_preservation():
    """15. Keyword-only candidates must receive RRF contribution and survive into candidate pool."""
    mock_vector = AsyncMock()
    mock_vector.search.return_value = []
    
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = [
        {"id": "chunk_kw_1", "title": "Laravel Guide", "score": 0.95, "content": "Laravel React PHP", "url": "https://webandcrafts.com/laravel"}
    ]
    
    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword)
    results = await hs.search("Laravel")
    
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_kw_1"
    assert results[0].keyword_score == 0.95
    assert results[0].vector_score == 0.0
    assert results[0].fusion_score > 0.0


@pytest.mark.anyio
async def test_16_vector_only_candidate_preservation():
    """16. Vector-only candidates must preserve vector score and receive RRF contribution."""
    mock_vector = AsyncMock()
    mock_vector.search.return_value = [
        {"id": "chunk_vec_1", "title": "AI Innovation", "score": 0.72, "content": "AI algorithms", "url": "https://webandcrafts.com/ai"}
    ]
    
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = []
    
    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword)
    results = await hs.search("AI Innovation")
    
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_vec_1"
    assert results[0].vector_score == 0.72
    assert results[0].keyword_score == 0.0
    assert results[0].fusion_score > 0.0


@pytest.mark.anyio
async def test_17_candidate_appearing_in_both_searches():
    """17. Candidate appearing in both searches receives combined RRF contribution."""
    chunk_shared = {"id": "chunk_shared_1", "title": "React Architecture", "content": "React and Node.js", "url": "https://webandcrafts.com/react"}
    
    mock_vector = AsyncMock()
    mock_vector.search.return_value = [{**chunk_shared, "score": 0.80}]
    
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = [{**chunk_shared, "score": 0.90}]
    
    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword)
    results = await hs.search("React Architecture")
    
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_shared_1"
    assert results[0].vector_score == 0.80
    assert results[0].keyword_score == 0.90
    # Combined RRF score for rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    assert results[0].fusion_score >= 0.030


@pytest.mark.anyio
async def test_18_stable_chunk_id_matching():
    """18. Stable chunk ID matches across disparate metadata representations."""
    mock_vector = AsyncMock()
    mock_vector.search.return_value = [{"id": "stable_123", "title": "Title V", "score": 0.70, "content": "C1", "url": "https://wac.co/1"}]
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = [{"id": "stable_123", "title": "Title K", "score": 0.85, "content": "C1", "url": "https://wac.co/1"}]

    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword)
    results = await hs.search("Query")
    assert len(results) == 1
    assert results[0].chunk_id == "stable_123"


@pytest.mark.anyio
async def test_19_keyword_score_preservation():
    """19. Keyword score is preserved accurately onto RetrievedChunk."""
    mock_vector = AsyncMock()
    mock_vector.search.return_value = []
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = [{"id": "kw_chunk", "score": 0.9563, "title": "T", "content": "C", "url": "U"}]

    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword)
    results = await hs.search("Test")
    assert results[0].keyword_score == 0.9563


@pytest.mark.anyio
async def test_20_rrf_score_calculation():
    """20. RRF score calculation follows reciprocal rank formula."""
    mock_vector = AsyncMock()
    mock_vector.search.return_value = [
        {"id": "c1", "score": 0.8, "title": "T1", "content": "C", "url": "U"},
        {"id": "c2", "score": 0.7, "title": "T2", "content": "C", "url": "U"},
    ]
    mock_keyword = AsyncMock()
    mock_keyword.search.return_value = []

    hs = HybridSearch(vector_search=mock_vector, keyword_search=mock_keyword, rrf_k=60)
    results = await hs.search("Test")
    assert round(results[0].fusion_score, 6) == round(1.0 / 61.0, 6)
    assert round(results[1].fusion_score, 6) == round(1.0 / 62.0, 6)


@pytest.mark.anyio
async def test_21_reranker_score_preservation():
    """21. Reranker updates reranked_score and score properly without discarding keyword score."""
    reranker = FusionReranker()
    chunk = RetrievedChunk(
        chunk_id="c_test",
        document_id="d1",
        title="Node.js & React Full Stack Development",
        heading_path=["Engineering"],
        content="We build microservices with Node.js and React frontend.",
        url="https://webandcrafts.com/engineering",
        canonical_url="https://webandcrafts.com/engineering",
        score=0.016,
        vector_score=0.0,
        keyword_score=0.94,
        fusion_score=0.016,
    )

    reranked = await reranker.rerank("What technology does WAC use? Node.js React", [chunk])
    assert len(reranked) == 1
    assert reranked[0].keyword_score == 0.94
    assert reranked[0].reranked_score >= 0.90
    assert reranked[0].score == reranked[0].reranked_score


@pytest.mark.anyio
async def test_22_evidence_sufficient_false_for_generic_slogan_evidence():
    """22. evidence_sufficient is False when retrieved chunks contain only generic marketing slogans."""
    rag_service = RAGService(min_relevance_score=0.50)
    generic_chunk = RetrievedChunk(
        chunk_id="c_slogan",
        document_id="d_slogan",
        title="WAC Beyond",
        heading_path=["Overview"],
        content="Intelligence, Elevated. AI + Insight. Story Reel.",
        url="https://webandcrafts.com",
        canonical_url="https://webandcrafts.com",
        score=0.75,
        vector_score=0.75,
        keyword_score=0.0,
        fusion_score=0.016,
    )

    sufficient = rag_service._evaluate_evidence_sufficiency(
        user_query="What technologies does WAC use?",
        rewritten_query="What technologies does WAC use?",
        chunks=[generic_chunk],
        confidence=0.75
    )
    assert sufficient is False


@pytest.mark.anyio
async def test_23_evidence_sufficient_true_for_actual_technology_evidence():
    """23. evidence_sufficient is True when retrieved chunks contain concrete technology tokens."""
    rag_service = RAGService(min_relevance_score=0.50)
    concrete_chunk = RetrievedChunk(
        chunk_id="c_tech",
        document_id="d_tech",
        title="WAC Tech Stack",
        heading_path=["Engineering"],
        content="Our technology stack includes React, Node.js, Python, Laravel, AWS, and MongoDB.",
        url="https://webandcrafts.com/tech",
        canonical_url="https://webandcrafts.com/tech",
        score=0.95,
        vector_score=0.70,
        keyword_score=0.94,
        fusion_score=0.030,
    )

    sufficient = rag_service._evaluate_evidence_sufficiency(
        user_query="What technologies does WAC use?",
        rewritten_query="What technologies does WAC use?",
        chunks=[concrete_chunk],
        confidence=0.95
    )
    assert sufficient is True


@pytest.mark.anyio
async def test_24_gemini_not_called_when_evidence_is_insufficient():
    """24. Gemini/LLM provider is NOT called when evidence_sufficient is False."""
    ai_service = AIService()
    mock_provider = AsyncMock()
    
    mock_rag_result = RAGResult(
        is_relevant=True,
        has_context=False,
        evidence_sufficient=False,
        context="",
        sources=[],
        retrieval_score=0.40,
        refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
    )
    
    with patch("app.ai.service.ProviderFactory.get_provider", return_value=mock_provider), \
         patch("app.services.rag_service.RAGService.get_grounded_context", AsyncMock(return_value=mock_rag_result)):
        response, rag_result = await ai_service.chat(
            provider="gemini",
            message="What technologies does WAC use?",
        )
        
        # Provider must NOT have been called
        mock_provider.generate.assert_not_called()
        assert "I couldn't find reliable information" in response.content


def test_25_ecommerce_query_expansion_contains_ecommerce_not_global_list():
    """25. Ecommerce query receives ecommerce expansion, not the global technology list."""
    query = "What ecommerce technologies does WAC use?"
    expanded = QueryRewriter.expand_for_retrieval(query)
    
    # Must contain ecommerce terms
    assert "WAC Commerce" in expanded
    assert "Adobe Commerce" in expanded
    assert "Shopify" in expanded
    assert "Magento" in expanded
    assert "WooCommerce" in expanded
    
    # Must NOT contain unrelated global tech list items
    assert "Laravel" not in expanded
    assert "Django" not in expanded
    assert "Flutter" not in expanded


def test_26_ecommerce_intent_classification():
    """26. Intent classifier accurately identifies ecommerce variants."""
    intent1 = QueryRewriter.detect_intent("What ecommerce technologies does WAC use?")
    assert intent1.category == "ECOMMERCE"
    
    intent2 = QueryRewriter.detect_intent("What e-commerce technologies does WAC use?")
    assert intent2.category == "ECOMMERCE"
    
    intent3 = QueryRewriter.detect_intent("Tell me about WAC Shopify store development.")
    assert intent3.category == "ECOMMERCE"


@pytest.mark.anyio
async def test_27_ecommerce_service_chunk_outranks_generic_laravel_blog():
    """27. Direct WAC ecommerce service chunk outranks generic Laravel blog chunk for ecommerce queries."""
    reranker = FusionReranker()
    
    ecommerce_service_chunk = RetrievedChunk(
        chunk_id="chunk_ecom_service",
        document_id="doc_ecom",
        title="#1 Ecommerce Website Development Services Company in India",
        heading_path=["E-commerce", "Next-gen Ecommerce Solutions with AI"],
        content="WAC introduces an innovative yet high-value Adobe Commerce extension known as 'WAC Commerce', that powers online stores with AI search and recommendations.",
        url="https://webandcrafts.com/services/e-commerce",
        canonical_url="https://webandcrafts.com/services/e-commerce",
        score=0.70,
        vector_score=0.70,
        keyword_score=0.85,
        fusion_score=0.016,
    )
    
    generic_laravel_blog = RetrievedChunk(
        chunk_id="chunk_laravel_blog",
        document_id="doc_laravel",
        title="Why Use Laravel in 2026: Features, Use Cases & Combinations",
        heading_path=["Engineering", "Laravel"],
        content="Laravel is a robust PHP web framework supporting MVC architecture and ORM database management.",
        url="https://webandcrafts.com/blog/laravel-features-use-cases",
        canonical_url="https://webandcrafts.com/blog/laravel-features-use-cases",
        score=0.80,
        vector_score=0.60,
        keyword_score=0.95,
        fusion_score=0.016,
    )
    
    reranked = await reranker.rerank(
        query="What ecommerce technologies does WAC use?",
        chunks=[generic_laravel_blog, ecommerce_service_chunk],
        top_k=2
    )
    
    assert len(reranked) == 2
    # Direct ecommerce service chunk must rank #1 over generic Laravel blog
    assert reranked[0].chunk_id == "chunk_ecom_service"


@pytest.mark.anyio
async def test_28_ecommerce_evidence_sufficiency_checks():
    """28. Evidence sufficiency is intent-aware: refuses generic blog, accepts ecommerce evidence."""
    rag_service = RAGService(min_relevance_score=0.50)
    
    generic_blog = RetrievedChunk(
        chunk_id="c_blog",
        document_id="d_blog",
        title="Why Use Laravel in 2026",
        heading_path=["Engineering"],
        content="Laravel provides elegant syntax and PHP tooling.",
        url="https://webandcrafts.com/blog/laravel",
        canonical_url="https://webandcrafts.com/blog/laravel",
        score=0.90,
        vector_score=0.60,
        keyword_score=0.90,
        fusion_score=0.016,
    )
    
    # Generic blog alone should fail evidence sufficiency for an ecommerce query
    assert rag_service._evaluate_evidence_sufficiency(
        user_query="What ecommerce technologies does WAC use?",
        rewritten_query="What ecommerce technologies does WAC use?",
        chunks=[generic_blog],
        confidence=0.90
    ) is False
    
    ecommerce_chunk = RetrievedChunk(
        chunk_id="c_ecom",
        document_id="d_ecom",
        title="#1 Ecommerce Website Development Services Company in India",
        heading_path=["E-commerce", "Technologies We Use For E-commerce", "React"],
        content="Build fast-loading and optimised web applications with rapid page rendering features from the constructive and interactive technology of React for ecommerce stores.",
        url="https://webandcrafts.com/services/e-commerce",
        canonical_url="https://webandcrafts.com/services/e-commerce",
        score=0.92,
        vector_score=0.70,
        keyword_score=0.90,
        fusion_score=0.020,
    )
    
    # Direct ecommerce chunk must satisfy evidence sufficiency
    assert rag_service._evaluate_evidence_sufficiency(
        user_query="What ecommerce technologies does WAC use?",
        rewritten_query="What ecommerce technologies does WAC use?",
        chunks=[ecommerce_chunk],
        confidence=0.92
    ) is True


def test_29_mobile_query_does_not_expand_into_ecommerce():
    """29. Mobile app query expands into mobile tech, not ecommerce platforms."""
    query = "What mobile technologies does WAC use?"
    expanded = QueryRewriter.expand_for_retrieval(query)
    
    assert "Flutter" in expanded or "iOS" in expanded or "Android" in expanded
    assert "Magento" not in expanded
    assert "Shopify" not in expanded


def test_30_explicit_tech_query_expansion():
    """30. Explicit technology question expands specifically for that tech."""
    query_react = "Does WAC use React?"
    expanded_react = QueryRewriter.expand_for_retrieval(query_react)
    assert "React" in expanded_react
    assert "Laravel" not in expanded_react
    
    query_node = "Does WAC use Node.js?"
    expanded_node = QueryRewriter.expand_for_retrieval(query_node)
    assert "Node.js" in expanded_node or "Node JS" in expanded_node
    assert "Flutter" not in expanded_node

