from app.ai.exceptions import AIException


class RAGException(AIException):
    """Base exception for RAG subsystem."""
    pass


class CrawlerException(RAGException):
    """Exception raised during crawling operation."""
    pass


class SSRFProtectionException(CrawlerException):
    """Exception raised when an invalid or disallowed domain/IP is target of crawl."""
    pass


class ExtractionException(RAGException):
    """Exception raised during HTML content parsing or cleaning."""
    pass


class ChunkingException(RAGException):
    """Exception raised during text chunking."""
    pass


class EmbeddingException(RAGException):
    """Exception raised during vector embedding generation."""
    pass


class RetrievalException(RAGException):
    """Exception raised during search retrieval or reranking."""
    pass


class GroundingException(RAGException):
    """Exception raised during response grounding verification."""
    pass


class RAGConfigurationException(RAGException):
    """Exception raised for invalid RAG configuration."""
    pass
