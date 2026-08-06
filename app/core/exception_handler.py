from fastapi import Request
from fastapi.responses import JSONResponse

from app.ai.exceptions import *


async def ai_exception_handler(
    request: Request,
    exc: AIException,
):

    status = 500
    code = "AI_ERROR"

    if isinstance(exc, RateLimitException):
        status = 429
        code = "RATE_LIMIT"

    elif isinstance(exc, InvalidAPIKeyException):
        status = 401
        code = "INVALID_API_KEY"

    elif isinstance(exc, ProviderUnavailableException):
        status = 503
        code = "PROVIDER_UNAVAILABLE"

    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": exc.message,
            },
        },
    )