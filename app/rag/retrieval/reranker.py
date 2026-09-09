from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.rag.models import RetrievedChunk
from app.rag.retrieval.query_rewriter import QueryRewriter, RetrievalIntent


class BaseReranker(ABC):
    """Abstract reranker interface for modular reranking implementations."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks for given query."""
        pass


class FusionReranker(BaseReranker):
    """
    Intent-aware reranker that combines:

    1. Base retrieval score (balanced vector + keyword signals)
    2. Intent matching & Direct Company Service Evidence Priority
    3. Title & Heading structural alignment
    4. Keyword matching in content (with stop-word filtering)
    5. Term coverage & density
    6. Freshness & URL diversity
    """

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
        "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
        "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
        "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
        "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "tell", "than", "that", "that's", "the", "their", "theirs",
        "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
        "they'll", "they're", "they've", "this", "those", "through", "to", "too",
        "under", "until", "up", "us", "use", "uses", "used", "using", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
        "what's", "when", "when's", "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:

        k = top_k or settings.RAG_TOP_K_FINAL

        if not chunks:
            return []

        # 1. Detect query intent
        intent = QueryRewriter.detect_intent(query)

        # 2. Extract meaningful terms filtering conversational stop words
        raw_terms = query.strip().split()
        query_terms: list[str] = []
        for t in raw_terms:
            cleaned = t.strip("?.,!;:'\"()[]{}").lower()
            if cleaned and cleaned not in self.STOP_WORDS and len(cleaned) >= 2:
                if cleaned not in query_terms:
                    query_terms.append(cleaned)

        if not query_terms and raw_terms:
            query_terms = [t.strip("?.,!;:'\"()[]{}").lower() for t in raw_terms if len(t) >= 2]

        reranked: list[RetrievedChunk] = []

        for chunk in chunks:
            v_score = chunk.vector_score or 0.0
            k_score = chunk.keyword_score or 0.0

            # Balanced base retrieval score combining vector similarity and keyword score
            if v_score > 0 and k_score > 0:
                base_score = (max(v_score, k_score) * 0.70) + (((v_score + k_score) / 2.0) * 0.30)
            elif v_score > 0:
                base_score = v_score
            elif k_score > 0:
                base_score = k_score
            else:
                base_score = min(1.0, (chunk.fusion_score or chunk.score or 0.0) * 61.0)

            title_lower = (chunk.title or "").lower()
            heading_str = " ".join(chunk.heading_path or []).lower()
            content_lower = (chunk.content or "").lower()
            url_lower = (chunk.url or "").lower()

            is_service_page = "/services/" in url_lower
            is_blog_page = "/blog/" in url_lower
            is_about_page = "/about" in url_lower or "/careers" in url_lower or "/contact" in url_lower

            title_matches = 0
            heading_matches = 0
            content_matches = 0
            matched_terms_count = 0

            for term in query_terms:
                in_t = term in title_lower
                in_h = term in heading_str
                in_c = term in content_lower

                if in_t:
                    title_matches += 1
                if in_h:
                    heading_matches += 1
                if in_c:
                    content_matches += 1
                if in_t or in_h or in_c:
                    matched_terms_count += 1

            # Standard structural and content boosts
            title_boost = min(title_matches * 0.10, 0.25)
            heading_boost = min(heading_matches * 0.08, 0.20)
            content_boost = min(content_matches * 0.05, 0.25)

            coverage_boost = 0.0
            if query_terms:
                coverage = matched_terms_count / len(query_terms)
                coverage_boost = coverage * 0.15

            # ------------------------------------------------------------------
            # Intent-aware Boost & Company Service Evidence Priority
            # ------------------------------------------------------------------
            intent_boost = 0.0
            intent_reasons: list[str] = []

            if intent.category == "ECOMMERCE":
                has_ecom_title = any(w in title_lower for w in ("ecommerce", "e-commerce", "shopify", "magento", "commerce", "store"))
                has_ecom_heading = any(w in heading_str for w in ("ecommerce", "e-commerce", "technologies we use for e-commerce", "shopify", "magento", "commerce"))
                has_ecom_content = any(w in content_lower for w in ("ecommerce", "e-commerce", "shopify", "magento", "woocommerce", "online store"))
                has_wac_commerce = "wac commerce" in content_lower or "adobe commerce" in content_lower or "wac commerce" in heading_str

                if has_ecom_title:
                    intent_boost += 0.15
                    intent_reasons.append("ecommerce_title_match")
                if has_ecom_heading:
                    intent_boost += 0.12
                    intent_reasons.append("ecommerce_heading_match")
                if is_service_page and ("e-commerce" in url_lower or "shopify" in url_lower or "ecommerce" in url_lower):
                    intent_boost += 0.15
                    intent_reasons.append("direct_ecommerce_service_page")
                if has_wac_commerce:
                    intent_boost += 0.12
                    intent_reasons.append("wac_adobe_commerce_content")
                elif has_ecom_content:
                    intent_boost += 0.08
                    intent_reasons.append("ecommerce_content_match")

                # De-prioritize purely educational tech blogs that do not focus on ecommerce
                if is_blog_page and not (has_ecom_title or has_ecom_heading or has_wac_commerce):
                    intent_boost -= 0.10
                    intent_reasons.append("generic_blog_penalty")

            elif intent.category == "DIGITAL_MARKETING":
                if "digital-marketing" in url_lower or is_service_page:
                    intent_boost += 0.15
                    intent_reasons.append("digital_marketing_service_page")
                if "marketing" in title_lower or "seo" in title_lower or "sem" in title_lower:
                    intent_boost += 0.12
                    intent_reasons.append("marketing_title_match")

            elif intent.category == "MOBILE_APP":
                if "mobile" in url_lower or "app" in url_lower:
                    intent_boost += 0.15
                    intent_reasons.append("mobile_service_page")
                if "mobile" in title_lower or "ios" in title_lower or "android" in title_lower:
                    intent_boost += 0.12
                    intent_reasons.append("mobile_title_match")

            elif intent.category == "EXPLICIT_TECH" and intent.technologies:
                target_tech = intent.technologies[0]
                has_company_grounding = is_service_page or any(
                    w in content_lower or w in heading_str or w in title_lower
                    for w in (
                        "wac", "webandcrafts", "web and crafts",
                        "our tech stack", "our technology stack", "technologies we use",
                        "our developers", "our engineers", "our team", "our services",
                        "our capabilities", "our expertise", "our solutions",
                        "we use", "we build", "we develop", "we provide", "we offer",
                        "we create", "we deliver", "we work with", "we specialize",
                        "hire", "at wac", "wac's", "developers at wac"
                    )
                )

                if has_company_grounding:
                    intent_boost += 0.15
                    intent_reasons.append("company_grounded_tech_evidence")
                elif is_blog_page and not has_company_grounding:
                    intent_boost -= 0.12
                    intent_reasons.append("generic_blog_without_company_grounding")

                if target_tech in title_lower:
                    intent_boost += 0.10
                    intent_reasons.append(f"explicit_{target_tech}_title")
                if target_tech in heading_str:
                    intent_boost += 0.08
                    intent_reasons.append(f"explicit_{target_tech}_heading")
                if target_tech in content_lower:
                    intent_boost += 0.06
                    intent_reasons.append(f"explicit_{target_tech}_content")

            elif intent.category == "TECHNOLOGY_GENERAL":
                has_company_grounding = is_service_page or any(
                    w in content_lower or w in heading_str or w in title_lower
                    for w in (
                        "wac", "webandcrafts", "web and crafts",
                        "our tech stack", "our technology stack", "technologies we use",
                        "our developers", "our engineers", "our team", "our services",
                        "our capabilities", "our expertise", "our solutions",
                        "we use", "we build", "we develop", "we provide", "we offer",
                        "we create", "we deliver", "we work with", "we specialize",
                        "hire", "at wac", "wac's", "developers at wac"
                    )
                )
                if has_company_grounding:
                    intent_boost += 0.15
                    intent_reasons.append("company_technology_stack_evidence")
                elif is_blog_page and not has_company_grounding:
                    intent_boost -= 0.12
                    intent_reasons.append("generic_tech_blog_penalty")

            elif intent.category == "SERVICES" and is_service_page:
                intent_boost += 0.15
                intent_reasons.append("service_page_priority")

            elif intent.category == "ABOUT_COMPANY" and is_about_page:
                intent_boost += 0.15
                intent_reasons.append("about_page_priority")

            # Bound intent boost to prevent artificial inflation
            bounded_intent_boost = round(min(0.35, max(-0.15, intent_boost)), 4)

            # Final composite score
            weighted_score = (
                (base_score * 0.45)
                + title_boost
                + heading_boost
                + content_boost
                + coverage_boost
                + bounded_intent_boost
            )
            final_score = round(min(1.0, max(0.0, max(base_score, weighted_score))), 4)

            chunk.reranked_score = final_score
            chunk.score = final_score

            reranked.append(chunk)

        # ---------------------------------------------
        # Highest relevance first, with freshness tie-breaking
        # ---------------------------------------------
        reranked.sort(
            key=lambda x: (
                x.score,
                x.updated_at.timestamp() if x.updated_at else (x.created_at.timestamp() if x.created_at else 0)
            ),
            reverse=True,
        )

        # Apply URL diversity: prioritize 1 chunk per unique URL in top results to avoid duplicate URL consumption
        diverse_results: list[RetrievedChunk] = []
        url_counts: dict[str, int] = {}
        deferred: list[RetrievedChunk] = []

        for c in reranked:
            url_key = c.canonical_url or c.url or c.chunk_id
            count = url_counts.get(url_key, 0)
            if count < 1:
                diverse_results.append(c)
                url_counts[url_key] = count + 1
            else:
                deferred.append(c)

        if len(diverse_results) < k:
            for c in deferred:
                if len(diverse_results) >= k:
                    break
                url_key = c.canonical_url or c.url or c.chunk_id
                count = url_counts.get(url_key, 0)
                if count < 2:
                    diverse_results.append(c)
                    url_counts[url_key] = count + 1

        if len(diverse_results) < k:
            diverse_results.extend(deferred[:k - len(diverse_results)])

        return diverse_results[:k]