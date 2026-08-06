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


class ChatResponse(BaseModel):
    conversation_id: str

    title: str

    response: str