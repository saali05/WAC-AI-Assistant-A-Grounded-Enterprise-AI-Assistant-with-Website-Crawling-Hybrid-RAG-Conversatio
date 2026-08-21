from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str | None = None

    provider: Literal["gemini", "groq"] = "gemini"

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


class SourceItem(BaseModel):
    title: str
    url: str
    heading: str | None = None
    score: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str

    title: str

    response: str

    sources: list[SourceItem] = []

    rag_used: bool = False

    retrieval_score: float | None = None