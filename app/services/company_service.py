from pathlib import Path
import re


class CompanyService:
    """
    Smart company knowledge loader.

    Loads only relevant sections from company.md
    """

    FILE = Path("app/knowledge/company.md")

    SECTION_KEYWORDS = {
        "COMPANY OVERVIEW": [
            "company",
            "about",
            "overview",
            "history",
            "wac",
            "web and craft",
        ],

        "COMPANY STATISTICS": [
            "projects",
            "employees",
            "experts",
            "countries",
            "clients",
            "statistics",
            "numbers",
        ],

        "CORE SERVICES": [
            "service",
            "services",
            "development",
            "design",
            "marketing",
            "branding",
        ],

        "TECHNOLOGY EXPERTISE": [
            "technology",
            "tech",
            "python",
            "django",
            "react",
            "fastapi",
            "cloud",
            "aws",
            "ai",
            "machine learning",
            "devops",
        ],

        "INDUSTRIES SERVED": [
            "industry",
            "industries",
            "healthcare",
            "retail",
            "education",
            "finance",
        ],

        "FEATURED CLIENTS": [
            "client",
            "customer",
            "customers",
        ],

        "FEATURED CASE STUDIES": [
            "case study",
            "portfolio",
            "project",
            "work",
        ],

        "CAREERS": [
            "career",
            "job",
            "internship",
            "vacancy",
            "opening",
            "hiring",
        ],

        "LOCATIONS": [
            "location",
            "office",
            "india",
            "usa",
            "address",
        ],

        "CONTACT INFORMATION": [
            "contact",
            "email",
            "phone",
            "support",
        ],

        "COMPANY VALUES": [
            "culture",
            "values",
            "mission",
            "vision",
            "purpose",
        ],
    }

    def __init__(self):

        self.document = self.FILE.read_text(
            encoding="utf-8"
        )

        self.sections = self._parse_sections()

    def _parse_sections(self):

        pattern = (
            r"={5,}\n"
            r"(.*?)\n"
            r"={5,}"
        )

        matches = list(
            re.finditer(
                pattern,
                self.document,
                re.DOTALL,
            )
        )

        sections = {}

        for index, match in enumerate(matches):

            title = match.group(1).strip()

            start = match.end()

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(self.document)
            )

            sections[title] = (
                self.document[start:end].strip()
            )

        return sections

    def get_context(
        self,
        question: str,
    ) -> str:

        question = question.lower()

        selected = []

        # Always include overview

        selected.append(
            "COMPANY OVERVIEW"
        )

        for section, keywords in self.SECTION_KEYWORDS.items():

            if any(
                keyword in question
                for keyword in keywords
            ):

                selected.append(section)

        selected = list(
            dict.fromkeys(selected)
        )

        context = []

        for section in selected:

            if section in self.sections:

                context.append(

                    f"{section}\n\n"

                    f"{self.sections[section]}"

                )

        return "\n\n".join(context)