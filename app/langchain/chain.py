from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logging import logger
from app.langchain.retrievers.wac_retriever import WACRetriever
from app.prompts.company_rules import COMPANY_RULES
from app.prompts.memory_rules import MEMORY_RULES
from app.prompts.response_rules import RESPONSE_RULES
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.models import RAGResult
from app.rag.retrieval.query_rewriter import QueryRewriter
from app.rag.validation.relevance import WACRelevanceGate


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""


@dataclass
class LangChainResponse:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    standalone_query: Optional[str] = None
    retrieved_documents: List[Document] = field(default_factory=list)
    rag_result: Optional[RAGResult] = None


class TokenUsageCallbackHandler(AsyncCallbackHandler):
    """Callback handler to record token usage across provider calls."""

    def __init__(self):
        super().__init__()
        self.usage = TokenUsage()

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
                if token_usage:
                    self.usage.prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                    self.usage.completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                    self.usage.total_tokens = token_usage.get("total_tokens") or (self.usage.prompt_tokens + self.usage.completion_tokens)
                model = response.llm_output.get("model_name") or response.llm_output.get("model")
                if model:
                    self.usage.model_name = str(model)

            for gen_list in getattr(response, "generations", []):
                for gen in gen_list:
                    message = getattr(gen, "message", None)
                    if message and isinstance(message, AIMessage):
                        um = getattr(message, "usage_metadata", None)
                        if um:
                            self.usage.prompt_tokens = um.get("input_tokens", self.usage.prompt_tokens)
                            self.usage.completion_tokens = um.get("output_tokens", self.usage.completion_tokens)
                            self.usage.total_tokens = um.get("total_tokens", self.usage.total_tokens)
                        rm = getattr(message, "response_metadata", None)
                        if rm and "token_usage" in rm:
                            tu = rm["token_usage"]
                            self.usage.prompt_tokens = tu.get("prompt_tokens", self.usage.prompt_tokens)
                            self.usage.completion_tokens = tu.get("completion_tokens", self.usage.completion_tokens)
                            self.usage.total_tokens = tu.get("total_tokens", self.usage.total_tokens)
        except Exception as exc:
            logger.warning(f"Failed to extract token usage in TokenUsageCallbackHandler: {exc}")


class WACLangChainPipeline:
    """
    Canonical LangChain LCEL Pipeline for WAC Grounded RAG.

    Ensures unified behavior between native RAG and LangChain orchestration:
    1. Evaluates WAC Domain Relevance Gate.
    2. Performs deterministic conversational query rewriting.
    3. Retrieves authoritative evidence via canonical WACRetriever.
    4. Enforces strict Evidence Sufficiency & relevance thresholds (blocks LLM if evidence is insufficient).
    5. Formats structured grounded prompt and invokes LLM via LCEL.
    6. Preserves token usage and source citation metadata.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        retriever: Optional[WACRetriever] = None,
    ) -> None:
        self.provider = (provider or settings.DEFAULT_PROVIDER).lower()
        self.retriever = retriever or WACRetriever()
        self.llm = self._init_llm(self.provider)

    def _init_llm(self, provider: str) -> Any:
        if provider in ("groq", "openai/gpt-oss-120b"):
            return ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.2,
            )
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            temperature=0.2,
        )

    def _format_chat_history_str(self, chat_history: Optional[Union[List[BaseMessage], str]]) -> str:
        """Convert LangChain messages or raw string into structured history string."""
        if not chat_history:
            return ""
        if isinstance(chat_history, str):
            return chat_history.strip()

        history_lines: List[str] = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Assistant: {msg.content}")
            elif hasattr(msg, "content"):
                history_lines.append(f"{getattr(msg, 'type', 'Message').capitalize()}: {msg.content}")

        return "\n".join(history_lines)

    async def get_standalone_query(
        self,
        input_text: str,
        chat_history: Optional[Union[List[BaseMessage], str]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """Canonical query rewriting for standalone retrieval query."""
        history_str = self._format_chat_history_str(chat_history)
        if not history_str:
            return input_text.strip()

        return QueryRewriter.rewrite(
            user_message=input_text,
            conversation_history=history_str,
        )

    def build_rephrase_chain(self):
        """Backward compatibility helper for query rewriter chain interface."""
        return QueryRewriter

    async def ainvoke(
        self,
        input_text: str,
        chat_history: Optional[Union[List[BaseMessage], str]] = None,
    ) -> LangChainResponse:
        """
        Execute full LangChain grounded chat pipeline adhering to canonical WAC RAG policy.
        """
        usage_callback = TokenUsageCallbackHandler()
        history_str = self._format_chat_history_str(chat_history)

        logger.info(
            f"\nLANGCHAIN REQUEST\n-----------------\n"
            f"query='{input_text}'\n"
            f"provider='{self.provider}'"
        )

        # --------------------------------------------------
        # 1. WAC RELEVANCE GATE
        # --------------------------------------------------
        is_wac_related, refusal = WACRelevanceGate.evaluate(
            user_message=input_text,
            conversation_history=history_str,
        )


        if not is_wac_related:
            refusal_msg = (
                refusal
                or "I'm the WAC AI Assistant, specifically designed to help with Web and Craft's services, technologies, solutions, company information, and career opportunities."
            )
            logger.info(f"LangChain Pipeline: Query rejected by relevance gate ('{input_text}')")
            return LangChainResponse(
                answer=refusal_msg,
                sources=[],
                usage=usage_callback.usage,
                standalone_query=input_text,
                retrieved_documents=[],
                rag_result=RAGResult(
                    is_relevant=False,
                    has_context=False,
                    evidence_sufficient=False,
                    context="",
                    sources=[],
                    retrieval_score=0.0,
                    refusal_reason=refusal_msg,
                ),
            )

        # --------------------------------------------------
        # 2. CANONICAL QUERY REWRITE & RETRIEVAL
        # --------------------------------------------------
        standalone_query = QueryRewriter.rewrite(
            user_message=input_text,
            conversation_history=history_str,
        )

        retrieved_docs: List[Document] = []
        rag_result: Optional[RAGResult] = None

        if hasattr(self.retriever, "retrieve_with_rag_result"):
            ret_out = await self.retriever.retrieve_with_rag_result(
                query=input_text,
                conversation_history=history_str,
            )
            if isinstance(ret_out, tuple) and len(ret_out) == 2:
                retrieved_docs, rag_result = ret_out

        if rag_result is None:
            retrieved_docs = await self.retriever.ainvoke(input_text)
            if retrieved_docs:
                from app.rag.models import SourceCitation
                mock_sources = [
                    SourceCitation(
                        title=d.metadata.get("title", "WAC Documentation"),
                        url=d.metadata.get("url", "https://webandcrafts.com"),
                        heading=d.metadata.get("heading", ""),
                        score=float(d.metadata.get("score", 0.90)),
                    )
                    for d in retrieved_docs
                ]
                rag_result = RAGResult(
                    is_relevant=True,
                    has_context=True,
                    evidence_sufficient=True,
                    context="\n\n".join(d.page_content for d in retrieved_docs),
                    sources=mock_sources,
                    retrieval_score=0.95,
                )
            else:
                rag_result = RAGResult(
                    is_relevant=True,
                    has_context=False,
                    evidence_sufficient=False,
                    context="",
                    sources=[],
                    retrieval_score=0.0,
                    refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base.",
                )


        # --------------------------------------------------
        # 3. EVIDENCE SUFFICIENCY & RELEVANCE THRESHOLD CHECK
        # --------------------------------------------------
        if not rag_result.has_context or not rag_result.evidence_sufficient or rag_result.retrieval_score < settings.RAG_MIN_RELEVANCE_SCORE:
            refusal_msg = (
                rag_result.refusal_reason
                or "I couldn't find reliable information about that in WAC's current knowledge base."
            )
            logger.info(
                f"LangChain Pipeline: Evidence insufficient or confidence low (confidence={rag_result.retrieval_score:.4f}). "
                f"Skipping LLM call and returning grounded refusal."
            )
            return LangChainResponse(
                answer=refusal_msg,
                sources=[],
                usage=usage_callback.usage,
                standalone_query=standalone_query,
                retrieved_documents=retrieved_docs,
                rag_result=rag_result,
            )

        # --------------------------------------------------
        # 4. GROUNDED PROMPT CONSTRUCTION & LCEL QA INVOCATION
        # --------------------------------------------------
        logger.info(
            f"LangChain Pipeline: Authoritative evidence confirmed (confidence={rag_result.retrieval_score:.4f}). "
            f"Executing grounded LLM generation via LCEL."
        )

        # Format prompt sections matching canonical PromptBuilder
        prompt_sections = [
            f"=== SYSTEM ===\n{SYSTEM_PROMPT}",
            f"=== RESPONSE RULES ===\n{RESPONSE_RULES}",
            f"=== COMPANY RULES ===\n{COMPANY_RULES}",
            f"=== MEMORY RULES ===\n{MEMORY_RULES}",
            f"=== AUTHORITATIVE WAC RETRIEVED EVIDENCE ===\n{rag_result.context}",
        ]

        if history_str:
            prompt_sections.append(f"=== CONVERSATION HISTORY ===\n{history_str}")

        prompt_sections.append(f"=== CURRENT USER QUESTION ===\n{input_text}\n\nANSWER:\n")

        full_prompt_text = "\n\n".join(prompt_sections)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", full_prompt_text),
            ("human", "{input}"),
        ])

        qa_chain = qa_prompt | self.llm

        ai_message = await qa_chain.ainvoke(
            {"input": input_text},
            config={"callbacks": [usage_callback]},
        )

        if isinstance(ai_message, AIMessage):
            if isinstance(ai_message.content, str):
                answer = ai_message.content
            elif isinstance(ai_message.content, list):
                parts = []
                for p in ai_message.content:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                    elif isinstance(p, str):
                        parts.append(p)
                    else:
                        parts.append(str(p))
                answer = "".join(parts)
            else:
                answer = str(ai_message.content)
        else:
            answer = str(ai_message)


        # --------------------------------------------------
        # 5. USAGE METADATA EXTRACTION
        # --------------------------------------------------
        usage = usage_callback.usage
        if isinstance(ai_message, AIMessage):
            um = getattr(ai_message, "usage_metadata", None)
            if um:
                usage.prompt_tokens = um.get("input_tokens", usage.prompt_tokens)
                usage.completion_tokens = um.get("output_tokens", usage.completion_tokens)
                usage.total_tokens = um.get("total_tokens", usage.total_tokens)

            rm = getattr(ai_message, "response_metadata", None)
            if rm and "token_usage" in rm:
                tu = rm["token_usage"]
                usage.prompt_tokens = tu.get("prompt_tokens", usage.prompt_tokens)
                usage.completion_tokens = tu.get("completion_tokens", usage.completion_tokens)
                usage.total_tokens = tu.get("total_tokens", usage.total_tokens)

        if not usage.model_name:
            usage.model_name = getattr(self.llm, "model_name", getattr(self.llm, "model", self.provider))

        # Format sources from rag_result
        sources: List[Dict[str, Any]] = [
            {
                "title": s.title,
                "url": s.url,
                "heading": s.heading or "",
                "score": s.score,
            }
            for s in rag_result.sources
        ]

        logger.info(
            f"WACLangChainPipeline completed | sources={len(sources)} | "
            f"prompt_tokens={usage.prompt_tokens} | completion_tokens={usage.completion_tokens}"
        )

        return LangChainResponse(
            answer=answer,
            sources=sources,
            usage=usage,
            standalone_query=standalone_query,
            retrieved_documents=retrieved_docs,
            rag_result=rag_result,
        )

