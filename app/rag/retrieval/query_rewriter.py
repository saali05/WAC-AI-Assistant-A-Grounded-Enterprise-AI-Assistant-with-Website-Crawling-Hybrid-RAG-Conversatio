import re


class QueryRewriter:
    """
    Rewrite conversational WAC follow-ups into standalone retrieval queries.

    The rewriter is intentionally deterministic.

    Rules:
        1. Explicit current topic -> keep current message.
        2. Pure continuation -> use previous user topic.
        3. Pronoun/contextual follow-up -> combine previous topic + current message.
        4. Standalone question -> keep current message.
    """

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

        # --------------------------------------------------------------
        # CASE 4:
        # Standalone query
        #
        # Do not use previous conversation context.
        # --------------------------------------------------------------

        return clean_user