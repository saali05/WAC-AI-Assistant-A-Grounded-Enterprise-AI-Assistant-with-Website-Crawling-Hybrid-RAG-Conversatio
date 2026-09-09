import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel

from app.langchain.retrievers.wac_retriever import WACRetriever
from app.langchain.chain import WACLangChainPipeline, TokenUsageCallbackHandler, LangChainResponse
from app.rag.models import RetrievedChunk, RAGResult, SourceCitation
from app.services.rag_service import RAGService


class DummyChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(
            content="WAC uses Adobe Commerce, Shopify, and Magento for enterprise ecommerce development.",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "dummy-chat-model"


@pytest.mark.anyio
async def test_wac_retriever_document_conversion():
    mock_hybrid_search = AsyncMock()
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        content="WAC builds AI solutions and enterprise software.",
        title="WAC AI Services",
        heading_path=["Services", "AI"],
        url="https://webandcrafts.com/services/ai",
        canonical_url="https://webandcrafts.com/services/ai",
        score=0.92,
        vector_score=0.95,
        keyword_score=0.88,
        fusion_score=0.90,
    )
    mock_hybrid_search.search.return_value = [chunk]

    retriever = WACRetriever(hybrid_search=mock_hybrid_search)
    docs = await retriever.ainvoke("AI services")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "WAC builds AI solutions and enterprise software."
    assert docs[0].metadata["id"] == "chunk_1"
    assert docs[0].metadata["title"] == "WAC AI Services"
    assert docs[0].metadata["heading"] == "Services > AI"
    assert docs[0].metadata["heading_path"] == ["Services", "AI"]
    assert docs[0].metadata["url"] == "https://webandcrafts.com/services/ai"
    assert docs[0].metadata["vector_score"] == 0.95
    assert docs[0].metadata["keyword_score"] == 0.88
    assert docs[0].metadata["fusion_score"] == 0.90
    assert docs[0].metadata["score"] >= 0.90
    assert docs[0].metadata["reranked_score"] >= 0.90



@pytest.mark.anyio
async def test_token_usage_callback_handler_metadata_extraction():
    handler = TokenUsageCallbackHandler()

    ai_message = AIMessage(
        content="Here are WAC's services.",
        usage_metadata={"input_tokens": 150, "output_tokens": 45, "total_tokens": 195},
    )
    gen = Generation(text="Here are WAC's services.", message=ai_message)
    llm_result = LLMResult(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195}, "model_name": "gemini-3.6-flash"},
    )

    await handler.on_llm_end(llm_result)

    assert handler.usage.prompt_tokens == 150
    assert handler.usage.completion_tokens == 45
    assert handler.usage.total_tokens == 195
    assert handler.usage.model_name == "gemini-3.6-flash"


@pytest.mark.anyio
async def test_wac_langchain_pipeline_construction():
    mock_retriever = AsyncMock(spec=WACRetriever)
    mock_retriever.ainvoke.return_value = [
        Document(
            page_content="Web and Crafts provides web development, AI, and cloud services.",
            metadata={"title": "WAC Services", "heading": "Services", "url": "https://webandcrafts.com/services", "score": 0.89},
        )
    ]

    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    assert pipeline.retriever == mock_retriever
    assert pipeline.provider == "gemini"

    rephrase_chain = pipeline.build_rephrase_chain()
    assert rephrase_chain is not None

    query = await pipeline.get_standalone_query("What services do you offer?", chat_history=[])
    assert query == "What services do you offer?"


@pytest.mark.anyio
async def test_wac_langchain_pipeline_end_to_end_mocked():
    mock_retriever = AsyncMock(spec=WACRetriever)
    mock_chunk = RetrievedChunk(
        chunk_id="chunk_101",
        document_id="doc_101",
        content="WAC offers full-stack digital transformation services.",
        title="WAC Solutions",
        heading_path=["Overview"],
        url="https://webandcrafts.com/solutions",
        canonical_url="https://webandcrafts.com/solutions",
        score=0.94,
        vector_score=0.95,
        keyword_score=0.90,
        fusion_score=0.92,
    )
    mock_rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        evidence_sufficient=True,
        context="WAC offers full-stack digital transformation services.",
        sources=[SourceCitation(title="WAC Solutions", url="https://webandcrafts.com/solutions", heading="Overview", score=0.94)],
        retrieved_chunks=[mock_chunk],
        retrieval_score=0.94,
    )
    mock_retriever.retrieve_with_rag_result.return_value = (
        [
            Document(
                page_content="WAC offers full-stack digital transformation services.",
                metadata={"id": "chunk_101", "title": "WAC Solutions", "heading": "Overview", "url": "https://webandcrafts.com/solutions", "score": 0.94},
            )
        ],
        mock_rag_result,
    )

    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    pipeline.llm = DummyChatModel()

    response = await pipeline.ainvoke("What services does WAC provide?")

    assert isinstance(response, LangChainResponse)
    assert "Adobe Commerce" in response.answer or "WAC" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0]["title"] == "WAC Solutions"
    assert response.sources[0]["heading"] == "Overview"
    assert response.sources[0]["url"] == "https://webandcrafts.com/solutions"
    assert response.usage.prompt_tokens == 120
    assert response.usage.completion_tokens == 30


@pytest.mark.anyio
async def test_langchain_refuses_out_of_domain():
    mock_retriever = AsyncMock(spec=WACRetriever)
    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    pipeline.llm = DummyChatModel()

    response = await pipeline.ainvoke("What is the capital of France?")

    assert isinstance(response, LangChainResponse)
    assert "specifically designed to help with Web and Craft" in response.answer or "WAC" in response.answer
    assert response.sources == []
    # Ensure retriever and LLM were not called
    assert mock_retriever.retrieve_with_rag_result.call_count == 0


@pytest.mark.anyio
async def test_langchain_evidence_insufficient_refusal():
    mock_retriever = AsyncMock(spec=WACRetriever)
    insufficient_rag_result = RAGResult(
        is_relevant=True,
        has_context=False,
        evidence_sufficient=False,
        context="",
        sources=[],
        retrieval_score=0.30,
        refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base.",
    )
    mock_retriever.retrieve_with_rag_result.return_value = ([], insufficient_rag_result)

    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    pipeline.llm = DummyChatModel()

    response = await pipeline.ainvoke("What is WAC's secret sauce?")

    assert "couldn't find reliable information" in response.answer
    assert response.sources == []
    assert response.usage.total_tokens == 0


@pytest.mark.anyio
async def test_langchain_conversation_followup_resolution():
    mock_retriever = AsyncMock(spec=WACRetriever)
    mock_chunk = RetrievedChunk(
        chunk_id="chunk_ecom",
        document_id="doc_ecom",
        content="WAC provides Adobe Commerce, Shopify, and Magento development services.",
        title="WAC E-commerce Services",
        heading_path=["Services", "E-commerce"],
        url="https://webandcrafts.com/services/e-commerce-development",
        canonical_url="https://webandcrafts.com/services/e-commerce-development",
        score=0.95,
        vector_score=0.96,
        keyword_score=0.92,
        fusion_score=0.94,
    )
    rag_result = RAGResult(
        is_relevant=True,
        has_context=True,
        evidence_sufficient=True,
        context="WAC provides Adobe Commerce, Shopify, and Magento development services.",
        sources=[SourceCitation(title="WAC E-commerce Services", url="https://webandcrafts.com/services/e-commerce-development", score=0.95)],
        retrieved_chunks=[mock_chunk],
        retrieval_score=0.95,
    )
    mock_retriever.retrieve_with_rag_result.return_value = (
        [Document(page_content=mock_chunk.content, metadata={"id": mock_chunk.chunk_id, "title": mock_chunk.title, "url": mock_chunk.url, "score": mock_chunk.score})],
        rag_result,
    )

    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    pipeline.llm = DummyChatModel()

    history = [
        HumanMessage(content="What ecommerce services does WAC provide?"),
        AIMessage(content="WAC provides end-to-end ecommerce development services."),
    ]

    response = await pipeline.ainvoke("Tell me more.", chat_history=history)

    assert isinstance(response, LangChainResponse)
    assert response.standalone_query is not None
    assert "ecommerce" in response.standalone_query.lower() or "tell me more" in response.standalone_query.lower()
    assert len(response.sources) >= 1

