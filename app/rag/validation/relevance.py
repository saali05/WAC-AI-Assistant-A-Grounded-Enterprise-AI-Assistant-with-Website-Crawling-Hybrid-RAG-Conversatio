import re


WAC_REFUSAL_MESSAGE = (
    "I'm the WAC AI Assistant, specifically designed to help with Web and Craft's "
    "services, technologies, solutions, company information, and career opportunities."
)

# WAC domain keywords and intent indicators
WAC_KEYWORDS = {
    "wac", "web and craft", "webandcrafts", "service", "services", "solution", "solutions",
    "technology", "tech", "ai", "machine learning", "cloud", "devops", "fastapi", "python",
    "react", "django", "career", "careers", "job", "jobs", "internship", "client", "clients",
    "portfolio", "case study", "location", "office", "contact", "about", "overview", "history",
    "project", "projects", "pricing", "pricing mode", "team", "leadership", "culture", "values"
}

# Out-of-domain keywords for immediate refusal
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(capital of|cook|recipe|biryani|weather in|who is the president|movie|sports|football|cricket)\b",
    r"^(what is python|what is javascript|how to cook|tell me a joke)\??$"
]


class WACRelevanceGate:
    """Lightweight relevance gate enforcing WAC-only domain scope."""

    @classmethod
    def evaluate(cls, user_message: str) -> tuple[bool, str | None]:
        """
        Evaluate if query is WAC-related.
        Returns (is_wac_related, refusal_message_or_None).
        """
        if not user_message or not user_message.strip():
            return False, WAC_REFUSAL_MESSAGE

        message_lower = user_message.strip().lower()

        # Check explicit out-of-domain patterns
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, message_lower):
                return False, WAC_REFUSAL_MESSAGE

        # Check WAC keywords
        words = set(re.findall(r"\b\w+\b", message_lower))
        if words.intersection(WAC_KEYWORDS):
            return True, None

        # Check general conversational greetings or short follow-ups
        short_greetings = {"hi", "hello", "hey", "good morning", "good evening", "greetings"}
        if message_lower in short_greetings or any(message_lower.startswith(g) for g in short_greetings):
            return True, None

        # If query has no WAC context or keywords, reject as out-of-domain
        if not any(kw in message_lower for kw in WAC_KEYWORDS):
            return False, WAC_REFUSAL_MESSAGE

        return True, None
