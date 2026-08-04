from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    provider: Literal["gemini", "groq"] = "gemini"

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message"
    )


class ChatResponse(BaseModel):
    response: str