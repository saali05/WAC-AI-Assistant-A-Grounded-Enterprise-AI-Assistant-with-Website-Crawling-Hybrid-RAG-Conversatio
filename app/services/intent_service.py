from enum import Enum
from app.rag.validation.relevance import WACRelevanceGate


class Intent(str, Enum):
    GREETING = "greeting"
    COMPANY = "company"
    SERVICE = "service"
    TECHNOLOGY = "technology"
    CAREER = "career"
    CONTACT = "contact"
    PROJECT = "project"
    GENERAL = "general"


class IntentService:

    INTENTS = {
        Intent.GREETING: [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "greetings",
        ],
        Intent.COMPANY: [
            "about",
            "company",
            "history",
            "overview",
            "wac",
            "web and craft",
        ],
        Intent.SERVICE: [
            "service",
            "services",
            "solution",
            "solutions",
        ],
        Intent.TECHNOLOGY: [
            "technology",
            "tech",
            "python",
            "react",
            "fastapi",
            "django",
            "cloud",
            "ai",
            "machine learning",
        ],
        Intent.CAREER: [
            "career",
            "job",
            "opening",
            "internship",
            "vacancy",
            "salary",
        ],
        Intent.CONTACT: [
            "contact",
            "email",
            "phone",
            "address",
            "office",
        ],
        Intent.PROJECT: [
            "project",
            "case study",
            "portfolio",
            "client",
        ],
    }

    def detect(self, question: str) -> Intent:
        if WACRelevanceGate.is_greeting(question):
            return Intent.GREETING

        question_lower = question.lower()
        for intent, keywords in self.INTENTS.items():
            if any(keyword in question_lower for keyword in keywords):
                return intent

        return Intent.GENERAL