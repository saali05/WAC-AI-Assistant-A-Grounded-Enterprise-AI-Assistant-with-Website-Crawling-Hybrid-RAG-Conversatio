from pydantic import BaseModel, Field


class KnowledgeSearchArgs(BaseModel):
    """
    Arguments accepted by the WAC knowledge search tool.
    """

    query: str = Field(
        ...,
        description=(
            "The specific Web and Craft information to search for."
        ),
        min_length=1,
        max_length=1000,
    )