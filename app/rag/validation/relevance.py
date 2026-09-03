import re


WAC_REFUSAL_MESSAGE = (
    "I'm the WAC AI Assistant, specifically designed to help with Web and Craft's biult by SALIM "
    "services, technologies, solutions, company information, and career opportunities."
)

WAC_GREETING_RESPONSE = (
    "Hello! Welcome to Web and Crafts (WAC) AI Assistant. "
    "How can I help you today with WAC's services, technologies, solutions, or company information?"
)

# Explicit WAC company markers
EXPLICIT_WAC_MARKERS = [
    "wac", "web and craft", "webandcrafts", "web & craft", "wac's",
    "your company", "your services", "your team", "your clients", "your projects",
    "your office", "the company", "our company"
]

# Explicit out-of-domain patterns for immediate refusal
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(capital of|cook|recipe|biryani|weather in|who is the president|movie|sports|football|cricket|tell me a joke|photosynthesis|napoleon)\b",
    r"^(how to cook|tell me a joke|write a poem|sing a song|what is photosynthesis)\??$"
]

# Generic technology / concept definition patterns (e.g., "What is React?")
GENERIC_DEFINITION_PATTERNS = [
    r"^(what|who|where|how|why)\s+(is|are|was|were|do|does)\s+(a\s+|an\s+|the\s+)?([a-z0-9\s\.\+#\-_]+)\??$"
]

# Affirmative / Continuation follow-up patterns for ongoing conversations
AFFIRMATIVE_FOLLOWUP_PATTERNS = [
    r"^(yes|yeah|yep|sure|ok|okay|please|tell me more|discuss|proceed|go ahead)\b",
    r"\b(discuss|tell me more|more info|more details|elaborate|go on|continue|specifics|tailored|options)\b"
]

# WAC-specific intent terms that indicate company services, capabilities, or details
WAC_INTENT_TERMS = {
    "service", "services", "solution", "solutions", "technology", "technologies",
    "career", "careers", "job", "jobs", "internship", "client", "clients",
    "portfolio", "case study", "location", "office", "contact", "about",
    "project", "projects", "pricing", "team", "leadership", "culture", "values"
}

# Greeting regex pattern matching hi, hii, hiii, hello, hey, good morning/afternoon/evening, greetings
GREETING_PATTERN = r"^(hi+|hello+|hey+|greetings|good\s*(morning|afternoon|evening))\b"


class WACRelevanceGate:
    """Conservative relevance gate enforcing WAC-only domain scope while supporting conversational follow-ups."""

    @classmethod
    def is_greeting(cls, user_message: str) -> bool:
        """Check if user message is a conversational greeting."""
        if not user_message or not user_message.strip():
            return False
        clean = re.sub(r"[^\w\s]", "", user_message.strip().lower()).strip()
        return bool(re.search(GREETING_PATTERN, clean))

    @classmethod
    def evaluate(cls, user_message: str, conversation_history: str = "") -> tuple[bool, str | None]:
        """
        Evaluate if query is WAC-related.
        Accepts optional conversation_history to support multi-turn conversational agreements.
        Returns (is_wac_related, refusal_message_or_None).
        """
        if not user_message or not user_message.strip():
            return False, WAC_REFUSAL_MESSAGE

        # 0. Greetings are valid WAC interactions
        if cls.is_greeting(user_message):
            return True, None

        message_lower = user_message.strip().lower()

        # 1. Check explicit out-of-domain patterns (Strict rejection even with history)
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, message_lower):
                return False, WAC_REFUSAL_MESSAGE

        # 2. Check for explicit WAC company markers
        has_wac_marker = any(marker in message_lower for marker in EXPLICIT_WAC_MARKERS)
        if has_wac_marker:
            return True, None

        # 3. Check for affirmative / continuation follow-up in ongoing conversation
        has_history = bool(conversation_history and conversation_history.strip())
        if has_history:
            for pattern in AFFIRMATIVE_FOLLOWUP_PATTERNS:
                if re.search(pattern, message_lower):
                    return True, None

        # 4. Check for generic definition queries without WAC markers (e.g., "What is React?", "What is Python?")
        for pattern in GENERIC_DEFINITION_PATTERNS:
            match = re.match(pattern, message_lower)
            if match:
                target = match.group(4).strip()
                # If target does not contain explicit WAC marker or company context, refuse as generic
                if not any(marker in target for marker in EXPLICIT_WAC_MARKERS):
                    return False, WAC_REFUSAL_MESSAGE

        # 5. If conversation history exists and message is a short conversational turn, allow continuation
        if has_history and len(message_lower.split()) <= 6:
            return True, None

        # 6. Check company capability/service intent terms
        words = set(re.findall(r"\b\w+\b", message_lower))
        if words.intersection(WAC_INTENT_TERMS):
            if any(term in message_lower for term in ["wac", "company", "services", "offer", "provide", "work with", "clients", "careers", "contact", "office", "projects"]):
                return True, None

        # Conservative default: reject non-WAC queries
        return False, WAC_REFUSAL_MESSAGE
