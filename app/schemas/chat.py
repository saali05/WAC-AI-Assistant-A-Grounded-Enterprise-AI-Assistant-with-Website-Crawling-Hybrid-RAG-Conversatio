from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request schema for chat messages.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User message",
        examples=["What is FastAPI?"],
    )


class ChatResponse(BaseModel):
    """
    Response schema returned by the chatbot.
    """

    response: str