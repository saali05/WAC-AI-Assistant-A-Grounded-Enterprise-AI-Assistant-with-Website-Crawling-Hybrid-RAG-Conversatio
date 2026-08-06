class AIException(Exception):
    """
    Base AI exception.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RateLimitException(AIException):
    pass


class InvalidAPIKeyException(AIException):
    pass


class ProviderUnavailableException(AIException):
    pass


class AIResponseException(AIException):
    pass