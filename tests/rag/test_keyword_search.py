from unittest.mock import AsyncMock, MagicMock
from bson import ObjectId
import pytest

from app.repositories.rag_repository import RAGChunkRepository


# ==============================================================================
# UNIT TESTS FOR TERM EXTRACTION & SCORING
# ==============================================================================

def test_extract_keyword_terms_technology_stack():
    query = (
        "What technologies does WAC use? WAC technology stack React Angular "
        "AngularJS Node.js Python PHP Laravel AWS Azure MongoDB Flutter Next.js Vue.js"
    )
    terms = RAGChunkRepository._extract_keyword_terms(query)
    
    assert "React" in terms
    assert "Node.js" in terms
    assert "Next.js" in terms
    assert "Vue.js" in terms
    assert "AngularJS" in terms
    assert "Laravel" in terms
    assert "Python" in terms
    assert "MongoDB" in terms
    assert "technologies" in terms or "technology" in terms
    
    # Conversational stop words must be filtered out
    assert "what" not in [t.lower() for t in terms]
    assert "does" not in [t.lower() for t in terms]
    assert "use" not in [t.lower() for t in terms]


def test_extract_keyword_terms_react():
    terms = RAGChunkRepository._extract_keyword_terms("React")
    assert terms == ["React"]


def test_extract_keyword_terms_nodejs():
    terms = RAGChunkRepository._extract_keyword_terms("Node.js")
    assert terms == ["Node.js"]


def test_extract_keyword_terms_laravel():
    terms = RAGChunkRepository._extract_keyword_terms("Laravel")
    assert terms == ["Laravel"]


def test_extract_keyword_terms_generic_wac_query():
    terms = RAGChunkRepository._extract_keyword_terms("What is WAC?")
    assert "WAC" in terms
    assert "what" not in [t.lower() for t in terms]
    assert "is" not in [t.lower() for t in terms]


def test_extract_keyword_terms_punctuation_handling():
    terms = RAGChunkRepository._extract_keyword_terms("Next.js, Vue.js; & Node.js! (AWS) [Python]...")
    assert "Next.js" in terms
    assert "Vue.js" in terms
    assert "Node.js" in terms
    assert "AWS" in terms
    assert "Python" in terms


def test_extract_keyword_terms_empty_query():
    assert RAGChunkRepository._extract_keyword_terms("") == []
    assert RAGChunkRepository._extract_keyword_terms("   ") == []


def test_calculate_keyword_score_field_weighting():
    terms = ["React", "Node.js"]
    
    doc_in_title = {
        "title": "React and Node.js Development Services",
        "heading_path": ["Services"],
        "content": "We build scalable web applications.",
    }
    
    doc_in_content_only = {
        "title": "Custom Software Development",
        "heading_path": ["Services"],
        "content": "Our stack includes React and Node.js.",
    }
    
    doc_unrelated = {
        "title": "Contact Us",
        "heading_path": ["About"],
        "content": "Reach out to our team.",
    }

    score_title = RAGChunkRepository._calculate_keyword_score(doc_in_title, terms)
    score_content = RAGChunkRepository._calculate_keyword_score(doc_in_content_only, terms)
    score_unrelated = RAGChunkRepository._calculate_keyword_score(doc_unrelated, terms)

    assert 0.0 <= score_title <= 1.0
    assert 0.0 <= score_content <= 1.0
    assert score_unrelated == 0.0
    assert score_title > score_content


# ==============================================================================
# ASYNC TESTS FOR KEYWORD_SEARCH (FALLBACK & TEXT SEARCH)
# ==============================================================================

@pytest.mark.anyio
async def test_keyword_search_fallback_behavior():
    mock_db = MagicMock()
    repo = RAGChunkRepository(db=mock_db)

    # Mock collection
    mock_collection = MagicMock()
    mock_db.rag_chunks = mock_collection

    # Simulate $text search throwing OperationFailure (no text index)
    cursor_text = MagicMock()
    cursor_text.to_list = AsyncMock(side_effect=Exception("text index required for $text query"))
    cursor_text.sort.return_value = cursor_text
    cursor_text.limit.return_value = cursor_text

    # Mock regex fallback returning candidates
    candidate_1 = {
        "_id": ObjectId("65f1a1b2c3d4e5f6a7b8c9d0"),
        "title": "WAC Frontend & Backend Stack",
        "heading_path": ["Technology", "Web"],
        "content": "We use React, Next.js, and Node.js for modern web applications.",
        "url": "https://webandcrafts.com/tech-stack",
        "status": "active",
    }
    candidate_2 = {
        "_id": ObjectId("65f1a1b2c3d4e5f6a7b8c9d1"),
        "title": "About WAC",
        "heading_path": ["About"],
        "content": "WAC is a technology company.",
        "url": "https://webandcrafts.com/about",
        "status": "active",
    }

    cursor_fallback = MagicMock()
    cursor_fallback.limit.return_value = cursor_fallback
    cursor_fallback.to_list = AsyncMock(return_value=[candidate_1, candidate_2])

    def find_mock(query, *args, **kwargs):
        if "$text" in query:
            return cursor_text
        return cursor_fallback

    mock_collection.find.side_effect = find_mock

    query = "What technologies does WAC use? React Node.js Next.js"
    results = await repo.keyword_search(query_str=query, top_k=5)

    assert len(results) == 2
    top_result = results[0]
    
    # Requirement #7: id, score, content, title, url
    assert "id" in top_result
    assert "score" in top_result
    assert "content" in top_result
    assert "title" in top_result
    assert "url" in top_result

    # Specific tech stack doc must outrank generic about doc
    assert top_result["id"] == "65f1a1b2c3d4e5f6a7b8c9d0"
    assert top_result["title"] == "WAC Frontend & Backend Stack"
    assert top_result["score"] > results[1]["score"]
    assert 0.0 <= top_result["score"] <= 1.0


@pytest.mark.anyio
async def test_keyword_search_empty_query():
    mock_db = MagicMock()
    repo = RAGChunkRepository(db=mock_db)
    
    results = await repo.keyword_search("")
    assert results == []
    
    results_whitespace = await repo.keyword_search("    ")
    assert results_whitespace == []
