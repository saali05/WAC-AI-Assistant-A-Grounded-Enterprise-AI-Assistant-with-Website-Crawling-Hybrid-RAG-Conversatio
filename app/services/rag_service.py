from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.rag.models import RAGResult
from app.rag.retrieval.context_builder import ContextBuilder
from app.rag.retrieval.hybrid_search import HybridSearch
from app.rag.retrieval.query_rewriter import QueryRewriter
from app.rag.retrieval.reranker import BaseReranker, FusionReranker
from app.rag.validation.relevance import WACRelevanceGate


class RAGService:
    """High-level RAG retrieval service orchestrating relevance gating, search, reranking, and context building."""

    KNOWN_TECH_TOKENS = {
        "react", "angular", "angularjs", "node.js", "nodejs", "python", "php",
        "laravel", "aws", "azure", "mongodb", "flutter", "next.js", "nextjs",
        "vue.js", "vuejs", "vue", "docker", "kubernetes", "graphql", "typescript",
        "javascript", "html", "css", "tailwind", "mysql", "postgresql", "redis",
        "drupal", "wordpress", "magento", "shopify", "salesforce", "figma", "adobe",
        "flutter", "swift", "kotlin", "ios", "android", "cloud", "api", "rest"
    }

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[BaseReranker] = None,
        min_relevance_score: Optional[float] = None
    ) -> None:
        self.hybrid_search = hybrid_search or HybridSearch()
        self.reranker = reranker or FusionReranker()
        self.min_relevance_score = min_relevance_score if min_relevance_score is not None else settings.RAG_MIN_RELEVANCE_SCORE

    @staticmethod
    def _has_company_relationship_evidence(chunk) -> bool:
        """
        Check if a chunk contains substantive evidence connecting WAC / the company
        to a technology, service, or capability, distinguishing company capability
        evidence from purely third-party educational/comparison articles.
        """
        url_lower = (getattr(chunk, "url", "") or "").lower()
        title_lower = (getattr(chunk, "title", "") or "").lower()
        heading_lower = " ".join(getattr(chunk, "heading_path", []) or []).lower()
        content_lower = (getattr(chunk, "content", "") or "").lower()
        combined = f"{title_lower} {heading_lower} {content_lower}"

        # 1. Authoritative company service, tech, or about URLs
        if any(seg in url_lower for seg in ("/services/", "/technology/", "/solutions/", "/careers/", "/about", "/contact", "/hire-", "/work/", "/portfolio/")):
            return True

        # 2. Explicit company capability & relationship markers in content
        company_markers = (
            "wac", "webandcrafts", "web and crafts",
            "our tech stack", "our technology stack", "technologies we use",
            "our developers", "our engineers", "our team", "our services",
            "our capabilities", "our expertise", "our solutions",
            "we use", "we build", "we develop", "we provide", "we offer",
            "we create", "we deliver", "we work with", "we specialize",
            "hire", "at wac", "wac's", "developers at wac"
        )
        return any(marker in combined for marker in company_markers)

    def _evaluate_evidence_sufficiency(
        self,
        user_query: str,
        rewritten_query: str,
        chunks: list,
        confidence: float
    ) -> bool:
        """
        Determine whether retrieved evidence actually contains substantive, intent-relevant
        information to answer the query, rather than vague marketing slogans or unrelated content.
        """
        if not chunks or confidence < self.min_relevance_score:
            return False

        intent = QueryRewriter.detect_intent(user_query)
        if intent.category == "GENERAL":
            intent = QueryRewriter.detect_intent(rewritten_query)

        # 1. E-COMMERCE queries: Must contain substantive ecommerce terms
        if intent.category == "ECOMMERCE":
            ecom_terms = {"ecommerce", "e-commerce", "wac commerce", "adobe commerce", "magento", "shopify", "woocommerce", "online store", "store", "commerce"}
            found_ecom = False
            for chunk in chunks[:5]:
                content_lower = (chunk.content or "").lower()
                title_lower = (chunk.title or "").lower()
                heading_lower = " ".join(chunk.heading_path or []).lower()
                combined = f"{title_lower} {heading_lower} {content_lower}"
                if any(t in combined for t in ecom_terms):
                    found_ecom = True
                    break
            if not found_ecom:
                logger.info("Evidence Sufficiency: E-commerce query lacked substantive ecommerce evidence in retrieved chunks.")
                return False

        # 2. EXPLICIT TECH queries (e.g. "Does WAC use React?"): Must contain target technology AND company relationship evidence
        elif intent.category == "EXPLICIT_TECH" and intent.technologies:
            target_tech = intent.technologies[0].lower()
            found_target = False
            has_company_grounding = False
            for chunk in chunks[:5]:
                content_lower = (chunk.content or "").lower()
                title_lower = (chunk.title or "").lower()
                heading_lower = " ".join(chunk.heading_path or []).lower()
                combined = f"{title_lower} {heading_lower} {content_lower}"
                if target_tech in combined:
                    found_target = True
                    if self._has_company_relationship_evidence(chunk):
                        has_company_grounding = True
                        break
            if not found_target:
                logger.info(f"Evidence Sufficiency: Explicit tech query lacked '{target_tech}' in retrieved chunks.")
                return False
            if not has_company_grounding:
                logger.info(f"Evidence Sufficiency: Explicit tech query found '{target_tech}' only in generic ungrounded articles without company relationship evidence.")
                return False

        # 3. GENERAL TECHNOLOGY queries: Must contain known technical tokens AND company relationship evidence
        elif intent.category == "TECHNOLOGY_GENERAL":
            found_tech = False
            has_company_grounding = False
            for chunk in chunks[:5]:
                content_lower = (chunk.content or "").lower()
                title_lower = (chunk.title or "").lower()
                heading_lower = " ".join(chunk.heading_path or []).lower()
                combined = f"{title_lower} {heading_lower} {content_lower}"
                if any(tech in combined for tech in self.KNOWN_TECH_TOKENS):
                    found_tech = True
                    if self._has_company_relationship_evidence(chunk):
                        has_company_grounding = True
                        break
            if not found_tech:
                logger.info("Evidence Sufficiency: Technology query lacked specific technology mentions in retrieved chunks.")
                return False
            if not has_company_grounding:
                logger.info("Evidence Sufficiency: Technology query lacked company-grounded technology stack evidence.")
                return False

        # 4. DIGITAL MARKETING queries: Must contain marketing terms
        elif intent.category == "DIGITAL_MARKETING":
            mkt_terms = {"digital marketing", "marketing", "seo", "sem", "social media", "ppc", "brand"}
            found_mkt = False
            for chunk in chunks[:5]:
                content_lower = (chunk.content or "").lower()
                title_lower = (chunk.title or "").lower()
                heading_lower = " ".join(chunk.heading_path or []).lower()
                combined = f"{title_lower} {heading_lower} {content_lower}"
                if any(t in combined for t in mkt_terms):
                    found_mkt = True
                    break
            if not found_mkt:
                logger.info("Evidence Sufficiency: Digital marketing query lacked marketing evidence in retrieved chunks.")
                return False

        # General check: Top chunk must have substantive content (>30 characters)
        top_chunk = chunks[0]
        if len((top_chunk.content or "").strip()) < 30 and len((top_chunk.title or "").strip()) < 10:
            logger.info("Evidence Sufficiency: Top chunk has insufficient text length.")
            return False

        return True

    async def get_grounded_context(
        self,
        user_message: str,
        conversation_history: str = ""
    ) -> RAGResult:
        """
        Execute full RAG pipeline:
        1. Evaluate WAC Relevance Gate
        2. Rewrite conversational query & generate intent-aware retrieval query
        3. Perform hybrid vector + keyword search
        4. Rerank retrieved chunks with company evidence priority
        5. Evaluate evidence sufficiency & apply relevance threshold
        6. Build context block and source citations
        """
        if not settings.RAG_ENABLED:
            return RAGResult(is_relevant=True, has_context=False, evidence_sufficient=False, context="", sources=[], retrieval_score=0.0)

        # 1. WAC Relevance Gate
        is_wac_related, refusal = WACRelevanceGate.evaluate(user_message, conversation_history=conversation_history)
        if not is_wac_related:
            logger.info(f"RAG Relevance Gate: Query rejected as out-of-domain ('{user_message}')")
            return RAGResult(
                is_relevant=False,
                has_context=False,
                evidence_sufficient=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason=refusal
            )

        # 2. Query Rewriting & Intent-Aware Expansion
        rewritten_query = QueryRewriter.rewrite(user_message, conversation_history)
        retrieval_intent = QueryRewriter.detect_intent(rewritten_query)
        retrieval_query = QueryRewriter.expand_for_retrieval(rewritten_query)

        logger.info(
            f"\nQUERY\n-----\nuser_query='{user_message}'\n"
            f"rewritten_query='{rewritten_query}'\n"
            f"intent='{retrieval_intent.category}'\n"
            f"retrieval_query='{retrieval_query}'"
        )

        # 3. Hybrid Search (retrieve candidate pool combining vector and keyword results)
        candidate_pool_limit = settings.RAG_TOP_K_VECTOR + settings.RAG_TOP_K_KEYWORD
        retrieved_chunks = await self.hybrid_search.search(retrieval_query, top_k=candidate_pool_limit)

        if not retrieved_chunks:
            logger.info(f"RAG Retrieval: No chunks retrieved for query '{retrieval_query}'")
            return RAGResult(
                is_relevant=True,
                has_context=False,
                evidence_sufficient=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
            )

        # 4. Reranking
        reranked_chunks = await self.reranker.rerank(
            retrieval_query, 
            retrieved_chunks, 
            top_k=settings.RAG_TOP_K_FINAL
        )

        if not reranked_chunks:
            return RAGResult(
                is_relevant=True,
                has_context=False,
                evidence_sufficient=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
            )

        # Log Final Reranked Chunks
        reranked_log_lines = ["\nFINAL RERANKED\n--------------"]
        for rank, chunk in enumerate(reranked_chunks, start=1):
            reranked_log_lines.append(
                f"rank={rank} chunk_id={chunk.chunk_id} title='{chunk.title}' "
                f"vector_score={chunk.vector_score or 0.0:.4f} keyword_score={chunk.keyword_score or 0.0:.4f} "
                f"fusion_score={chunk.fusion_score or 0.0:.4f} reranked_score={chunk.reranked_score or 0.0:.4f} "
                f"url='{chunk.url}'"
            )
        logger.info("\n".join(reranked_log_lines))

        top_chunk = reranked_chunks[0]

        # Calculate normalized retrieval confidence score (0.0 to 1.0)
        vector_val = top_chunk.vector_score if top_chunk.vector_score is not None else 0.0
        reranked_val = top_chunk.reranked_score if top_chunk.reranked_score is not None else top_chunk.score
        keyword_val = top_chunk.keyword_score if top_chunk.keyword_score is not None else 0.0
        fusion_val = top_chunk.fusion_score if top_chunk.fusion_score is not None else 0.0

        if vector_val > 0 and keyword_val > 0:
            confidence = min(1.0, (vector_val * 0.30 + reranked_val * 0.70))
        elif keyword_val > 0:
            confidence = min(1.0, (keyword_val * 0.40 + reranked_val * 0.60))
        elif vector_val > 0:
            confidence = min(1.0, (vector_val * 0.30 + reranked_val * 0.70))
        else:
            confidence = min(1.0, reranked_val)

        confidence = round(confidence, 4)
        top_chunk.retrieval_confidence = confidence

        # 5. Evidence Sufficiency Evaluation
        evidence_sufficient = self._evaluate_evidence_sufficiency(
            user_query=user_message,
            rewritten_query=rewritten_query,
            chunks=reranked_chunks,
            confidence=confidence,
        )

        logger.info(
            f"\nEVIDENCE\n--------\nevidence_sufficient={evidence_sufficient}\nconfidence={confidence:.4f}"
        )

        # Relevance Threshold & Evidence Sufficiency Check
        if not evidence_sufficient or confidence < self.min_relevance_score:
            logger.info(
                f"RAG Threshold/Sufficiency: evidence_sufficient={evidence_sufficient}, "
                f"confidence={confidence:.4f} (min={self.min_relevance_score})"
            )
            return RAGResult(
                is_relevant=True,
                has_context=False,
                evidence_sufficient=False,
                context="",
                sources=[],
                retrieval_score=confidence,
                refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
            )

        # 6. Context Building
        context_str, sources = ContextBuilder.build_context_and_sources(reranked_chunks)

        context_log_lines = [f"\nCONTEXT BUILT\n-------------\nsource_count={len(reranked_chunks)}"]
        for rank, chunk in enumerate(reranked_chunks, start=1):
            heading = " > ".join(chunk.heading_path) if chunk.heading_path else "N/A"
            preview = (chunk.content or "")[:120].replace("\n", " ")
            context_log_lines.append(
                f"rank={rank} title='{chunk.title}' heading='{heading}' url='{chunk.url}' preview='{preview}...'"
            )
        logger.info("\n".join(context_log_lines))

        return RAGResult(
            is_relevant=True,
            has_context=True,
            evidence_sufficient=True,
            context=context_str,
            sources=sources,
            retrieved_chunks=reranked_chunks,
            retrieval_score=confidence
        )

