from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class MessageCreate(BaseModel):

    conversation_id: str

    role: Literal["user", "assistant"]

    provider: Literal["gemini", "groq"]

    content: str


class MessageResponse(MessageCreate):

    id: str

    created_at: datetime