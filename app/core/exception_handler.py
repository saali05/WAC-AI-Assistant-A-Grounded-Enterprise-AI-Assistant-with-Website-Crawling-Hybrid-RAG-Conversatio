from fastapi import Request
from fastapi.responses import JSONResponse

from app.ai.exceptions import *


async def ai_exception_handler(
    request: Request,
    exc: AIException,
):

    status = 500
    code = "AI_ERROR"

    error_payload = {
        "code": code,
        "message": exc.message,
    }

    if isinstance(exc, RateLimitException):
        status = 429
        code = "RATE_LIMITED"
        error_payload["code"] = code
        if exc.provider:
            error_payload["provider"] = exc.provider
        if exc.model:
            error_payload["model"] = exc.model
        if exc.retry_after_seconds is not None:
            error_payload["retry_after_seconds"] = exc.retry_after_seconds
        if exc.limit_type:
            error_payload["limit_type"] = exc.limit_type

    elif isinstance(exc, InvalidAPIKeyException):
        status = 401
        code = "INVALID_API_KEY"
        error_payload["code"] = code

    elif isinstance(exc, ProviderUnavailableException):
        status = 503
        code = "PROVIDER_UNAVAILABLE"
        error_payload["code"] = code

    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "error": error_payload,
        },
    )