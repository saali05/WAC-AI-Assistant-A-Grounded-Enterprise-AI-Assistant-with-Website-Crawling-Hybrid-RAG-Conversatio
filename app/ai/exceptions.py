class AIException(Exception):
    """
    Base AI exception.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RateLimitException(AIException):
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        retry_after_seconds: int | None = None,
        limit_type: str | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type



class InvalidAPIKeyException(AIException):
    pass


class ProviderUnavailableException(AIException):
    pass


class AIResponseException(AIException):
    pass