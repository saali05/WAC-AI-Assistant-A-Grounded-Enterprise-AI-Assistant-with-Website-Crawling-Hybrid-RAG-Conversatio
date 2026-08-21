import pytest
from app.rag.validation.relevance import WACRelevanceGate, WAC_REFUSAL_MESSAGE


def test_wac_relevance_gate_accepted_queries():
    wac_queries = [
        "What services does WAC provide?",
        "What AI capabilities does WAC offer?",
        "What technologies does WAC use?",
        "Tell me about WAC careers and job openings",
        "Where are WAC offices located?",
    ]
    for q in wac_queries:
        is_relevant, refusal = WACRelevanceGate.evaluate(q)
        assert is_relevant is True
        assert refusal is None


def test_wac_relevance_gate_refused_queries():
    unrelated_queries = [
        "What is the capital of France?",
        "How do I cook biryani?",
        "What is Python?",
    ]
    for q in unrelated_queries:
        is_relevant, refusal = WACRelevanceGate.evaluate(q)
        assert is_relevant is False
        assert refusal == WAC_REFUSAL_MESSAGE
