from dataclasses import dataclass


@dataclass
class PromptSection:
    """
    Represents a reusable prompt section.
    """

    title: str
    content: str

    def render(self) -> str:

        if not self.content.strip():
            return ""

        return f"""
==================================================
{self.title}
==================================================

{self.content.strip()}
"""