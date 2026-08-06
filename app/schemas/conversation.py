from datetime import datetime

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: str

    title: str

    created_at: datetime

    updated_at: datetime

class RenameConversationRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )