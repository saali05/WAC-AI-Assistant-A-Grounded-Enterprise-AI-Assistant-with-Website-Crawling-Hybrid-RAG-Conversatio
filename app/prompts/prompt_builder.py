from app.ai.schemas import AIRequest

from app.prompts.prompt_section import PromptSection

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.response_rules import RESPONSE_RULES
from app.prompts.company_rules import COMPANY_RULES
from app.prompts.memory_rules import MEMORY_RULES


class PromptBuilder:
    """
    Builds the complete AI prompt from reusable sections.
    """

    @staticmethod
    def build(request: AIRequest) -> str:

        sections = [

            PromptSection(
                title="SYSTEM",
                content=SYSTEM_PROMPT,
            ),

            PromptSection(
                title="RESPONSE RULES",
                content=RESPONSE_RULES,
            ),

            PromptSection(
                title="COMPANY RULES",
                content=COMPANY_RULES,
            ),

            PromptSection(
                title="MEMORY RULES",
                content=MEMORY_RULES,
            ),

            PromptSection(
                title="WEB AND CRAFT KNOWLEDGE",
                content=request.company_context,
            ),

            PromptSection(
                title="SESSION MEMORY",
                content=request.conversation_history,
            ),

            PromptSection(
                title="CURRENT USER QUESTION",
                content=request.user_message,
            ),
        ]

        rendered_sections = []

        for section in sections:
            rendered = section.render()
            if rendered:
                rendered_sections.append(rendered)

        rendered_sections.append(
            "\nANSWER:\n"
        )

        return "\n".join(rendered_sections)