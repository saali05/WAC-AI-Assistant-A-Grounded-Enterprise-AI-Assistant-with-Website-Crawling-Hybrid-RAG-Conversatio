import re


WAC_REFUSAL_MESSAGE = (
    "I'm the WAC AI Assistant, specifically designed to help with Web and Craft's "
    "services, technologies, solutions, company information, and career opportunities."
)

# Explicit WAC company markers
EXPLICIT_WAC_MARKERS = [
    "wac", "web and craft", "webandcrafts", "web & craft", "wac's",
    "your company", "your services", "your team", "your clients", "your projects",
    "your office", "the company", "our company"
]

# Explicit out-of-domain patterns for immediate refusal
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(capital of|cook|recipe|biryani|weather in|who is the president|movie|sports|football|cricket|tell me a joke)\b",
    r"^(how to cook|tell me a joke|write a poem|sing a song)\??$"
]

# Generic technology / concept definition patterns (e.g., "What is React?")
GENERIC_DEFINITION_PATTERNS = [
    r"^(what|who|where|how|why)\s+(is|are|was|were|do|does)\s+(a\s+|an\s+|the\s+)?([a-z0-9\s\.\+#\-_]+)\??$"
]

# WAC-specific intent terms that indicate company services, capabilities, or details
WAC_INTENT_TERMS = {
    "service", "services", "solution", "solutions", "technology", "technologies",
    "career", "careers", "job", "jobs", "internship", "client", "clients",
    "portfolio", "case study", "location", "office", "contact", "about",
    "project", "projects", "pricing", "team", "leadership", "culture", "values"
}


class WACRelevanceGate:
    """Conservative relevance gate enforcing WAC-only domain scope."""

    @classmethod
    def evaluate(cls, user_message: str) -> tuple[bool, str | None]:
        """
        Evaluate if query is WAC-related.
        Returns (is_wac_related, refusal_message_or_None).
        """
        if not user_message or not user_message.strip():
            return False, WAC_REFUSAL_MESSAGE

        message_lower = user_message.strip().lower()

        # 1. Check explicit out-of-domain patterns
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, message_lower):
                return False, WAC_REFUSAL_MESSAGE

        # 2. Check for explicit WAC company markers
        has_wac_marker = any(marker in message_lower for marker in EXPLICIT_WAC_MARKERS)

        if has_wac_marker:
            return True, None

        # 3. Check for generic definition queries without WAC markers (e.g., "What is React?", "What is Python?")
        for pattern in GENERIC_DEFINITION_PATTERNS:
            match = re.match(pattern, message_lower)
            if match:
                target = match.group(4).strip()
                # If target does not contain explicit WAC marker or company context, refuse as generic
                if not any(marker in target for marker in EXPLICIT_WAC_MARKERS):
                    return False, WAC_REFUSAL_MESSAGE

        # 4. Check company capability/service intent terms
        words = set(re.findall(r"\b\w+\b", message_lower))
        if words.intersection(WAC_INTENT_TERMS):
            # Check if query asks specifically about company/services rather than standalone generic terms
            # E.g. "digital marketing", "ecommerce", "careers", "company initiatives"
            if any(term in message_lower for term in ["wac", "company", "services", "offer", "provide", "work with", "clients", "careers", "contact", "office", "projects"]):
                return True, None

        # 5. Check conversational greetings
        short_greetings = {"hi", "hello", "hey", "good morning", "good evening", "greetings"}
        if message_lower in short_greetings or any(message_lower.startswith(g) for g in short_greetings):
            return True, None

        # Conservative default: reject non-WAC queries
        return False, WAC_REFUSAL_MESSAGE

