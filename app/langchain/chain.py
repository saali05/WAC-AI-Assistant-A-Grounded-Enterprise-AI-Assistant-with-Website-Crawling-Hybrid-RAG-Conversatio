from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logging import logger
from app.langchain.retrievers.wac_retriever import WACRetriever
from app.prompts.system_prompt import SYSTEM_PROMPT


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
            logger.warning("Failed to extract token usage in TokenUsageCallbackHandler: %s", exc)


class WACLangChainPipeline:
    """
    LangChain LCEL Pipeline for WAC Grounded RAG.
    
    Includes:
    1. History-aware query rewriter step (LCEL) before WACRetriever.
    2. Grounded QA generation chain with WAC system prompt.
    3. Token usage metadata extraction from output AIMessage and callback collectors.
    """

    REPHRASE_PROMPT_SYSTEM = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. If the user responds with an affirmation, agreement, "
        "or follow-up (such as 'yes', 'sure', 'tell me more', 'yes i would like to discuss', "
        "'proceed', 'go ahead'), formulate a detailed standalone question based on the specific "
        "topic, service, or suggestion offered in the assistant's previous message. "
        "Do NOT answer the question, just reformulate it into a clear standalone search query."
    )

    QA_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\nUse the following retrieved context to answer the user's question:\n\n{context}"

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

    def build_rephrase_chain(self):
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", self.REPHRASE_PROMPT_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return rephrase_prompt | self.llm | StrOutputParser()

    async def get_standalone_query(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        if not chat_history:
            return input_text

        rephrase_chain = self.build_rephrase_chain()
        try:
            standalone = await rephrase_chain.ainvoke(
                {"input": input_text, "chat_history": chat_history},
                config={"callbacks": callbacks} if callbacks else None,
            )
            return standalone.strip() if standalone else input_text
        except Exception as exc:
            logger.warning("History-aware query rewriter failed, falling back to original query: %s", exc)
            return input_text

    async def ainvoke(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> LangChainResponse:
        chat_history = chat_history or []
        usage_callback = TokenUsageCallbackHandler()

        # Step 1: History-Aware Query Rewriter
        standalone_query = await self.get_standalone_query(
            input_text=input_text,
            chat_history=chat_history,
            callbacks=[usage_callback],
        )

        logger.info(
            "WACLangChainPipeline step 1 (Rewriter) completed | original='%s' | standalone='%s'",
            input_text,
            standalone_query,
        )

        # Step 2: Retrieve Documents via WACRetriever
        retrieved_docs: List[Document] = await self.retriever.ainvoke(standalone_query)

        # Build context string
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant WAC knowledge found."

        # Step 3: QA Answer Generation via LCEL
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.QA_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = qa_prompt | self.llm

        ai_message: AIMessage = await qa_chain.ainvoke(
            {
                "input": input_text,
                "context": context_str,
                "chat_history": chat_history,
            },
            config={"callbacks": [usage_callback]},
        )

        answer = ai_message.content if isinstance(ai_message, AIMessage) else str(ai_message)

        # Step 4: Extract Usage & Response Metadata from output AIMessage before string parsing
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

        # Format sources
        sources: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
            meta = doc.metadata or {}
            heading = meta.get("heading")
            if not heading and meta.get("heading_path"):
                hp = meta.get("heading_path")
                heading = " > ".join(hp) if isinstance(hp, list) else str(hp)
            sources.append({
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "heading": heading or "",
                "score": meta.get("score", 0.0),
            })

        logger.info(
            "WACLangChainPipeline completed | docs=%d | prompt_tokens=%d | completion_tokens=%d",
            len(retrieved_docs),
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return LangChainResponse(
            answer=answer,
            sources=sources,
            usage=usage,
            standalone_query=standalone_query,
            retrieved_documents=retrieved_docs,
        )
