from dataclasses import dataclass, field
import re
from typing import Optional


@dataclass
class RetrievalIntent:
    category: str
    entities: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    preferred_terms: list[str] = field(default_factory=list)


class QueryRewriter:
    """
    Rewrite conversational WAC follow-ups into standalone retrieval queries
    and generate intent-aware retrieval query expansions.

    The rewriter is intentionally deterministic.

    Rules:
        1. Explicit current topic -> keep current message.
        2. Pure continuation -> use previous user topic.
        3. Pronoun/contextual follow-up -> combine previous topic + current message.
        4. Standalone question -> keep current message.
    """

    # ------------------------------------------------------------------
    # Intent category definitions & controlled expansion terms
    # ------------------------------------------------------------------

    INTENT_EXPANSIONS = {
        "ECOMMERCE": (
            "WAC ecommerce ecommerce development ecommerce technologies "
            "WAC Commerce Adobe Commerce Magento Shopify WooCommerce "
            "React AI search product recommendations order management analytics"
        ),
        "DIGITAL_MARKETING": (
            "WAC digital marketing SEO SEM social media performance marketing "
            "content marketing PPC brand strategy marketing campaigns"
        ),
        "MOBILE_APP": (
            "WAC mobile app development iOS Android Flutter React Native "
            "Swift Kotlin cross platform applications"
        ),
        "CLOUD": (
            "WAC cloud services AWS Azure Google Cloud DevOps Kubernetes "
            "Docker cloud infrastructure migration microservices"
        ),
        "AI": (
            "WAC AI artificial intelligence machine learning generative AI "
            "LLM data analytics predictive solutions computer vision"
        ),
        "WEB_DEVELOPMENT": (
            "WAC custom web development frontend backend full stack "
            "React Next.js Node.js Laravel Python PHP web applications"
        ),
        "TECHNOLOGY_GENERAL": (
            "WAC technology stack web development React Angular AngularJS "
            "Node.js Python PHP Laravel AWS Azure MongoDB Flutter Next.js Vue.js"
        ),
        "SERVICES": (
            "WAC services custom software ecommerce web development "
            "mobile apps UI UX design digital marketing cloud solutions"
        ),
        "ABOUT_COMPANY": (
            "Webandcrafts WAC company overview leadership founders "
            "technology innovation digital transformation Kochi Calicut India"
        ),
        "CAREERS": (
            "WAC careers job opportunities hiring software engineer "
            "developer recruitment work culture"
        ),
        "CONTACT": (
            "WAC contact get in touch email address phone inquiry support office location"
        ),
    }

    # Intent detection patterns
    ECOMMERCE_PATTERNS = (
        r"\b(e-?commerce|online\s+stores?|commerce\s+platforms?|magento|shopify|woocommerce|adobe\s+commerce|wac\s+commerce|bigcommerce|shopping\s+cart|online\s+shopping)\b",
    )

    DIGITAL_MARKETING_PATTERNS = (
        r"\b(digital\s+marketing|seo|sem|social\s+media|content\s+marketing|ppc|performance\s+marketing|search\s+engine\s+optimi[zs]ation|brand\s+strategy)\b",
    )

    MOBILE_PATTERNS = (
        r"\b(mobile\s+app|mobile\s+apps|mobile\s+development|ios|android|react\s+native|swift|kotlin|mobile\s+applications?)\b",
    )

    CLOUD_PATTERNS = (
        r"\b(cloud|aws|azure|google\s+cloud|gcp|devops|kubernetes|docker|microservices|serverless)\b",
    )

    AI_PATTERNS = (
        r"\b(artificial\s+intelligence|machine\s+learning|generative\s+ai|gen\s*ai|llm|deep\s+learning|nlp|computer\s+vision|data\s+analytics)\b",
    )

    WEB_PATTERNS = (
        r"\b(web\s+development|web\s+applications?|frontend|backend|full\s*stack|cms\s+development|custom\s+web)\b",
    )

    SERVICES_PATTERNS = (
        r"\b(services|offerings|solutions|what\s+do\s+you\s+do|what\s+does\s+wac\s+offer|what\s+does\s+wac\s+provide|capabilities)\b",
    )

    ABOUT_PATTERNS = (
        r"\b(about\s+wac|who\s+is\s+wac|what\s+is\s+wac|about\s+webandcrafts|founders?|leadership|ceo|history\s+of\s+wac|wac\s+office|wac\s+location)\b",
    )

    CAREERS_PATTERNS = (
        r"\b(careers?|jobs?|hiring|openings?|vacancies|work\s+at\s+wac|recruitment|apply\s+at\s+wac)\b",
    )

    CONTACT_PATTERNS = (
        r"\b(contact|reach\s+out|email\s+address|phone\s+number|get\s+in\s+touch|office\s+address|inquiry)\b",
    )

    SPECIFIC_TECH_TOKENS = {
        "react": "React React.js WAC React React development React.js development",
        "react.js": "React React.js WAC React React development React.js development",
        "nodejs": "Node.js Node JS WAC Node.js Node.js development backend",
        "node.js": "Node.js Node JS WAC Node.js Node.js development backend",
        "laravel": "Laravel PHP framework WAC Laravel backend web development",
        "python": "Python Django FastAPI Flask WAC Python AI backend development",
        "php": "PHP Laravel WAC PHP backend web development",
        "angular": "Angular AngularJS WAC Angular frontend web application",
        "angularjs": "Angular AngularJS WAC AngularJS frontend web application",
        "vue": "Vue.js Vue WAC Vue frontend web application",
        "vue.js": "Vue.js Vue WAC Vue.js frontend web application",
        "next.js": "Next.js Nextjs WAC Next.js React frontend framework",
        "nextjs": "Next.js Nextjs WAC Next.js React frontend framework",
        "flutter": "Flutter cross platform mobile app development WAC Flutter",
        "aws": "AWS Amazon Web Services cloud infrastructure WAC AWS",
        "azure": "Microsoft Azure cloud services WAC Azure",
        "mongodb": "MongoDB NoSQL database WAC MongoDB",
        "shopify": "Shopify Shopify Plus ecommerce development store migration WAC Shopify",
        "magento": "Magento Adobe Commerce ecommerce store migration WAC Magento",
    }

    TECHNOLOGY_QUERY_PATTERNS = (
        r"\b(what|which)\s+(technology|technologies|tech|stack|tech\s+stack|frameworks?|tools?)\b",
        r"\b(technology|technologies|tech)\s+stack\b",
        r"\b(technologies|technology|tech)\s+(does|do|is|are|used|use)\b",
        r"\bwhat\s+tech\b",
        r"\bwhich\s+tech\b",
        r"\btech\s+does\b",
        r"\btech\s+do\b",
        r"\b(what|which)\s+.*?\b(technology|technologies|tech|stack)\b",
        r"\b(wac|webandcrafts)('s)?\s+(tech|technology|technologies|stack)\b",
    )

    @classmethod
    def detect_intent(cls, query: str) -> RetrievalIntent:
        """
        Deterministically classify query retrieval intent into a structured RetrievalIntent object.
        """
        normalized = cls._normalize(query)

        # 1. Check E-commerce intent
        if any(re.search(p, normalized) for p in cls.ECOMMERCE_PATTERNS):
            preferred = ["ecommerce", "e-commerce", "WAC Commerce", "Adobe Commerce", "Magento", "Shopify", "WooCommerce", "React"]
            return RetrievalIntent(category="ECOMMERCE", entities=["WAC", "ecommerce"], preferred_terms=preferred)

        # 2. Check Digital Marketing intent
        if any(re.search(p, normalized) for p in cls.DIGITAL_MARKETING_PATTERNS):
            preferred = ["digital marketing", "SEO", "SEM", "social media", "content marketing", "PPC", "performance marketing"]
            return RetrievalIntent(category="DIGITAL_MARKETING", entities=["WAC", "digital marketing"], preferred_terms=preferred)

        # 3. Check Mobile App intent
        if any(re.search(p, normalized) for p in cls.MOBILE_PATTERNS):
            preferred = ["mobile app", "iOS", "Android", "Flutter", "React Native", "Swift", "Kotlin"]
            return RetrievalIntent(category="MOBILE_APP", entities=["WAC", "mobile"], preferred_terms=preferred)

        # 4. Check Cloud intent
        if any(re.search(p, normalized) for p in cls.CLOUD_PATTERNS):
            preferred = ["cloud", "AWS", "Azure", "DevOps", "Kubernetes", "Docker"]
            return RetrievalIntent(category="CLOUD", entities=["WAC", "cloud"], preferred_terms=preferred)

        # 5. Check AI intent
        if any(re.search(p, normalized) for p in cls.AI_PATTERNS):
            preferred = ["AI", "artificial intelligence", "machine learning", "generative AI", "LLM", "data analytics"]
            return RetrievalIntent(category="AI", entities=["WAC", "AI"], preferred_terms=preferred)

        # 6. Check Specific Technology query (e.g. "Does WAC use React?", "Does WAC use Node.js?")
        matched_specific: list[str] = []
        for tech_token, expansion in cls.SPECIFIC_TECH_TOKENS.items():
            pattern = rf"(?<![\w#+.-]){re.escape(tech_token)}(?![\w#+.-])"
            if re.search(pattern, normalized):
                matched_specific.append(tech_token)

        if matched_specific and not any(re.search(p, normalized) for p in cls.SERVICES_PATTERNS):
            # If asking about a specific technology without general services query
            return RetrievalIntent(
                category="EXPLICIT_TECH",
                entities=["WAC"],
                technologies=matched_specific,
                preferred_terms=[cls.SPECIFIC_TECH_TOKENS[t] for t in matched_specific]
            )

        # 7. Check General Technology intent (e.g. "What technology does WAC use?")
        if any(re.search(p, normalized) for p in cls.TECHNOLOGY_QUERY_PATTERNS):
            preferred = ["React", "Angular", "Node.js", "Python", "PHP", "Laravel", "AWS", "Azure", "MongoDB", "Flutter", "Next.js", "Vue.js"]
            return RetrievalIntent(category="TECHNOLOGY_GENERAL", entities=["WAC", "technology"], preferred_terms=preferred)

        # 8. Check Web Development intent
        if any(re.search(p, normalized) for p in cls.WEB_PATTERNS):
            preferred = ["web development", "custom web applications", "frontend", "backend", "full stack"]
            return RetrievalIntent(category="WEB_DEVELOPMENT", entities=["WAC", "web development"], preferred_terms=preferred)

        # 9. Check Careers intent
        if any(re.search(p, normalized) for p in cls.CAREERS_PATTERNS):
            preferred = ["careers", "jobs", "hiring", "openings", "vacancies", "recruitment"]
            return RetrievalIntent(category="CAREERS", entities=["WAC", "careers"], preferred_terms=preferred)

        # 10. Check Contact intent
        if any(re.search(p, normalized) for p in cls.CONTACT_PATTERNS):
            preferred = ["contact", "email", "phone", "address", "inquiry", "get in touch"]
            return RetrievalIntent(category="CONTACT", entities=["WAC", "contact"], preferred_terms=preferred)

        # 11. Check About Company intent
        if any(re.search(p, normalized) for p in cls.ABOUT_PATTERNS):
            preferred = ["Webandcrafts", "WAC", "leadership", "founders", "company overview", "history"]
            return RetrievalIntent(category="ABOUT_COMPANY", entities=["WAC", "company"], preferred_terms=preferred)

        # 12. Check General Services intent
        if any(re.search(p, normalized) for p in cls.SERVICES_PATTERNS):
            preferred = ["services", "solutions", "offerings", "capabilities"]
            return RetrievalIntent(category="SERVICES", entities=["WAC", "services"], preferred_terms=preferred)

        return RetrievalIntent(category="GENERAL", entities=["WAC"], preferred_terms=[])

    # ------------------------------------------------------------------
    # Pure continuation patterns
    # ------------------------------------------------------------------

    PURE_CONTINUATION_PATTERNS = (
        r"^(yes|yeah|yep|sure|ok|okay|please)\.?$",
        r"^(tell me more|more info|more information|more details)\.?$",
        r"^(go on|continue|elaborate|proceed|go ahead)\.?$",
        r"^(discuss|let'?s discuss)\.?$",
    )

    # ------------------------------------------------------------------
    # Explicit topic patterns
    # ------------------------------------------------------------------

    EXPLICIT_TOPIC_PATTERNS = (
        r"^tell me more about .+",
        r"^more about .+",
        r"^details about .+",
        r"^information about .+",
        r"^tell me about .+",
        r"^give me more information about .+",
        r"^give me more details about .+",
        r"^explain .+",
        r"^describe .+",
        r"^what are .+",
        r"^what is .+",
        r"^what are the .+",
        r"^what is the .+",
        r"^how does .+",
        r"^how do .+",
        r"^why does .+",
        r"^why do .+",
        r"^where is .+",
        r"^where are .+",
        r"^who is .+",
        r"^who are .+",
    )

    # ------------------------------------------------------------------
    # Affirmative prefixes
    # ------------------------------------------------------------------

    AFFIRMATIVE_PREFIXES = (
        "yes ",
        "yeah ",
        "yep ",
        "sure ",
        "ok ",
        "okay ",
        "please ",
    )

    # ------------------------------------------------------------------
    # Generic continuation phrases
    # ------------------------------------------------------------------

    GENERIC_CONTINUATION_PHRASES = (
        "i want to know more",
        "i would like to know more",
        "i'd like to know more",
        "i want to learn more",
        "i would like to learn more",
        "i'd like to learn more",
        "i want more information",
        "i would like more information",
        "i'd like more information",
        "i want more details",
        "i would like more details",
        "i'd like more details",
        "tell me more",
        "know more",
        "learn more",
        "more about it",
        "more about that",
        "more details",
        "more information",
        "go ahead",
        "continue",
        "proceed",
        "elaborate",
        "discuss",
        "please continue",
        "please elaborate",
        "please tell me more",
    )

    # ------------------------------------------------------------------
    # Contextual words
    # ------------------------------------------------------------------

    CONTEXTUAL_WORDS = {
        "it",
        "that",
        "this",
        "them",
        "those",
        "these",
        "they",
        "he",
        "she",
        "there",
        "here",
    }

    @classmethod
    def is_technology_query(cls, query: str) -> bool:
        """
        Detect questions asking about WAC's technologies or technology stack.
        """
        intent = cls.detect_intent(query)
        if intent.category in ("TECHNOLOGY_GENERAL", "EXPLICIT_TECH"):
            return True

        normalized = cls._normalize(query)
        return any(
            re.search(pattern, normalized)
            for pattern in cls.TECHNOLOGY_QUERY_PATTERNS
        )

    @classmethod
    def _is_technology_query(cls, query: str) -> bool:
        return cls.is_technology_query(query)

    @classmethod
    def expand_for_retrieval(cls, query: str) -> str:
        """
        Intent-aware retrieval expansion:
        Adds controlled, intent-specific terms to the retrieval query without
        contaminating unrelated domains.
        """
        if not query or not query.strip():
            return query

        intent = cls.detect_intent(query)
        base_query = query.strip()

        if intent.category == "EXPLICIT_TECH" and intent.technologies:
            # Expand only with specific target tech terms
            expansions = [cls.SPECIFIC_TECH_TOKENS.get(t, t) for t in intent.technologies]
            expansion_str = " ".join(expansions)
            return f"{base_query} {expansion_str}"

        if intent.category in cls.INTENT_EXPANSIONS:
            expansion_str = cls.INTENT_EXPANSIONS[intent.category]
            return f"{base_query} {expansion_str}"

        if cls.is_technology_query(query):
            return f"{base_query} {cls.INTENT_EXPANSIONS['TECHNOLOGY_GENERAL']}"

        return base_query

    # ------------------------------------------------------------------
    # User-message extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_messages(conversation_history: str) -> list[str]:
        """
        Extract only User messages from conversation history.
        """

        if not conversation_history:
            return []

        history_lines = [
            line.strip()
            for line in conversation_history.splitlines()
            if line.strip()
        ]

        user_messages: list[str] = []

        for line in history_lines:
            match = re.match(
                r"^User:\s*(.+)$",
                line,
                re.IGNORECASE,
            )

            if match:
                user_messages.append(
                    match.group(1).strip()
                )

        return user_messages

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(message: str) -> str:
        """
        Normalize whitespace and lowercase the message.

        Used only for classification.
        The original user message is preserved as the retrieval query.
        """

        return re.sub(
            r"\s+",
            " ",
            (message or "").strip().lower(),
        )

    # ------------------------------------------------------------------
    # Remove trailing punctuation
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_trailing_punctuation(message: str) -> str:
        """
        Remove trailing punctuation.

        Examples:

            "tell me more?"
                -> "tell me more"

            "yes i want to know more!"
                -> "yes i want to know more"
        """

        return re.sub(
            r"[.!?]+$",
            "",
            message.strip(),
        ).strip()

    # ------------------------------------------------------------------
    # Pure continuation detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pure_continuation(message: str) -> bool:
        """
        Detect whether the current message contains no new topic.

        Examples:

            "yes"
            "tell me more"
            "yes i want to know more"
            "yeah i would like to learn more"
            "sure, tell me more"
        """

        normalized = QueryRewriter._normalize(message)

        if not normalized:
            return False

        # --------------------------------------------------------------
        # Exact continuation phrases
        # --------------------------------------------------------------

        exact_phrases = {
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "please",
            "go ahead",
            "proceed",
            "continue",
            "go on",
            "tell me more",
            "more info",
            "more information",
            "more details",
            "elaborate",
            "discuss",
            "let's discuss",
            "lets discuss",
        }

        if normalized.rstrip(".!?") in exact_phrases:
            return True

        # --------------------------------------------------------------
        # Regex continuation patterns
        # --------------------------------------------------------------

        normalized_no_punctuation = (
            QueryRewriter._strip_trailing_punctuation(
                normalized
            )
        )

        for pattern in QueryRewriter.PURE_CONTINUATION_PATTERNS:
            if re.match(
                pattern,
                normalized_no_punctuation,
                re.IGNORECASE,
            ):
                return True

        # --------------------------------------------------------------
        # Affirmative + generic continuation
        #
        # Examples:
        #
        # "yes i want to know more"
        # "yes i would like to learn more"
        # "sure tell me more"
        # "okay i want more details"
        # --------------------------------------------------------------

        for prefix in QueryRewriter.AFFIRMATIVE_PREFIXES:
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix):].strip()

                # Remove optional comma after "yes", "sure", etc.
                remainder = re.sub(
                    r"^,\s*",
                    "",
                    remainder,
                )

                remainder = (
                    QueryRewriter._strip_trailing_punctuation(
                        remainder
                    )
                )

                for phrase in (
                    QueryRewriter.GENERIC_CONTINUATION_PHRASES
                ):
                    if remainder == phrase:
                        return True

                    if remainder.startswith(
                        phrase + " "
                    ):
                        return True

        return False

    # ------------------------------------------------------------------
    # Explicit topic detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_explicit_topic(message: str) -> bool:
        """
        Detect whether the current message contains its own meaningful
        topic.

        Examples:

            "what about digital marketing?"
                -> True

            "how about ecommerce?"
                -> True

            "tell me more about cloud services"
                -> True

            "what about it?"
                -> False

            "how about that?"
                -> False

            "yes, tell me more about digital marketing"
                -> True
        """

        normalized = QueryRewriter._normalize(message)

        if not normalized:
            return False

        normalized = (
            QueryRewriter._strip_trailing_punctuation(
                normalized
            )
        )

        # --------------------------------------------------------------
        # Contextual-only words
        #
        # These do not represent a real new topic.
        # --------------------------------------------------------------

        contextual_only = {
            "it",
            "that",
            "this",
            "them",
            "those",
            "these",
            "they",
            "there",
            "here",
        }

        # --------------------------------------------------------------
        # Direct "what about ..." / "how about ..."
        # --------------------------------------------------------------

        match = re.match(
            r"^(what about|how about)\s+(.+)$",
            normalized,
            re.IGNORECASE,
        )

        if match:
            topic = match.group(2).strip()

            if topic in contextual_only:
                return False

            return True

        # --------------------------------------------------------------
        # Other explicit topic patterns
        # --------------------------------------------------------------

        if any(
            re.match(
                pattern,
                normalized,
                re.IGNORECASE,
            )
            for pattern in QueryRewriter.EXPLICIT_TOPIC_PATTERNS
        ):
            return True

        # --------------------------------------------------------------
        # Affirmative + explicit topic
        #
        # Examples:
        #
        # "yes, tell me more about digital marketing"
        # "sure, tell me about ecommerce"
        # "okay, what about digital marketing"
        # --------------------------------------------------------------

        affirmative_prefixes = (
            "yes ",
            "yeah ",
            "yep ",
            "sure ",
            "ok ",
            "okay ",
        )

        for prefix in affirmative_prefixes:
            if not normalized.startswith(prefix):
                continue

            remainder = normalized[len(prefix):].strip()

            # Remove optional comma.
            remainder = re.sub(
                r"^,\s*",
                "",
                remainder,
            )

            # ----------------------------------------------------------
            # "yes, what about digital marketing"
            # ----------------------------------------------------------

            match = re.match(
                r"^(what about|how about)\s+(.+)$",
                remainder,
                re.IGNORECASE,
            )

            if match:
                topic = match.group(2).strip()

                if topic not in contextual_only:
                    return True

            # ----------------------------------------------------------
            # "yes, tell me more about digital marketing"
            # ----------------------------------------------------------

            if any(
                re.match(
                    pattern,
                    remainder,
                    re.IGNORECASE,
                )
                for pattern in QueryRewriter.EXPLICIT_TOPIC_PATTERNS
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Contextual follow-up detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_contextual_followup(message: str) -> bool:
        """
        Detect messages that depend on previous conversation context.

        Examples:

            "what about it?"
            "how does that work?"
            "tell me about them"
            "what are those?"
        """

        normalized = QueryRewriter._normalize(message)

        words = set(
            re.findall(
                r"\b\w+\b",
                normalized,
            )
        )

        return bool(
            words.intersection(
                QueryRewriter.CONTEXTUAL_WORDS
            )
        )

    # ------------------------------------------------------------------
    # Main rewrite function
    # ------------------------------------------------------------------

    @staticmethod
    def rewrite(
        user_message: str,
        conversation_history: str = "",
    ) -> str:
        """
        Convert a conversational user message into a standalone
        retrieval query.

        Decision order:

            1. Empty message
            2. No history
            3. Extract previous user message
            4. Explicit current topic
            5. Pure continuation
            6. Contextual/pronoun follow-up
            7. Standalone query
        """

        # --------------------------------------------------------------
        # Step 1: Clean current message
        # --------------------------------------------------------------

        clean_user = (user_message or "").strip()

        if not clean_user:
            return clean_user

        # --------------------------------------------------------------
        # Step 2: No history
        # --------------------------------------------------------------

        if not conversation_history or not conversation_history.strip():
            return clean_user

        # --------------------------------------------------------------
        # Step 3: Extract previous User messages
        # --------------------------------------------------------------

        previous_user_messages = (
            QueryRewriter._extract_user_messages(
                conversation_history
            )
        )

        # --------------------------------------------------------------
        # Step 4: Remove accidental duplication of current message
        #
        # ChatService may include the current user message inside
        # conversation_history.
        # --------------------------------------------------------------

        current_normalized = (
            QueryRewriter._normalize(clean_user)
        )

        previous_user_messages = [
            message
            for message in previous_user_messages
            if QueryRewriter._normalize(message)
            != current_normalized
        ]

        # No previous user query available.
        if not previous_user_messages:
            return clean_user

        previous_user_query = previous_user_messages[-1]

        # --------------------------------------------------------------
        # CASE 1:
        # Explicit current topic
        #
        # Examples:
        #
        # "what about digital marketing?"
        # "how about ecommerce?"
        # "yes, tell me more about digital marketing"
        # "What technologies does WAC use?"
        #
        # Keep the current message.
        # --------------------------------------------------------------

        if QueryRewriter._has_explicit_topic(clean_user):
            return clean_user

        # --------------------------------------------------------------
        # CASE 2:
        # Pure continuation
        #
        # Examples:
        #
        # "yes"
        # "tell me more"
        # "yes i want to know more"
        # "yes i would like to learn more"
        #
        # Use the previous user topic.
        # --------------------------------------------------------------

        if QueryRewriter._is_pure_continuation(clean_user):
            return previous_user_query

        # --------------------------------------------------------------
        # CASE 3:
        # Contextual/pronoun follow-up
        #
        # Examples:
        #
        # "what about it?"
        # "how does that work?"
        # "tell me about them"
        #
        # Combine previous topic + current message.
        # --------------------------------------------------------------

        if QueryRewriter._is_contextual_followup(clean_user):
            return f"{previous_user_query} {clean_user}"

        return clean_user