from enum import Enum


class Intent(str, Enum):

    COMPANY = "company"

    SERVICE = "service"

    TECHNOLOGY = "technology"

    CAREER = "career"

    CONTACT = "contact"

    PROJECT = "project"

    GREETING = "greeting"

    GENERAL = "general"


class IntentService:

    INTENTS = {

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

        Intent.GREETING: [

            "hi",

            "hello",

            "hey",

            "good morning",

            "good evening",

        ],

    }

    def detect(self, question: str) -> Intent:

        question = question.lower()

        for intent, keywords in self.INTENTS.items():

            if any(keyword in question for keyword in keywords):

                return intent

        return Intent.GENERAL