import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel

from app.langchain.retrievers.wac_retriever import WACRetriever
from app.langchain.chain import WACLangChainPipeline, TokenUsageCallbackHandler, LangChainResponse
from app.rag.models import RetrievedChunk


class DummyChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(
            content="WAC provides digital transformation services.",
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
        url="https://webandcrafts.com/ai",
        canonical_url="https://webandcrafts.com/ai",
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
    assert docs[0].metadata["title"] == "WAC AI Services"
    assert docs[0].metadata["heading"] == "Services > AI"
    assert docs[0].metadata["url"] == "https://webandcrafts.com/ai"


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
    mock_retriever.ainvoke.return_value = [
        Document(
            page_content="WAC offers full-stack digital transformation services.",
            metadata={"title": "WAC Solutions", "heading": "Overview", "url": "https://webandcrafts.com/solutions", "score": 0.94},
        )
    ]

    pipeline = WACLangChainPipeline(provider="gemini", retriever=mock_retriever)
    pipeline.llm = DummyChatModel()

    response = await pipeline.ainvoke("What services does WAC provide?")

    assert isinstance(response, LangChainResponse)
    assert "digital transformation" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0]["title"] == "WAC Solutions"
    assert response.sources[0]["heading"] == "Overview"
    assert response.usage.prompt_tokens == 120
    assert response.usage.completion_tokens == 30
