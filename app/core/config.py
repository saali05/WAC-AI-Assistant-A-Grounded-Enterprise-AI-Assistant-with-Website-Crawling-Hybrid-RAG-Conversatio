from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool

    # AI Providers
    DEFAULT_PROVIDER: str

    GEMINI_API_KEY: str
    GEMINI_MODEL: str
    GEMINI_LIVE_MODEL: str = "gemini-3.1-flash-live-preview"

    GROQ_API_KEY: str
    GROQ_MODEL: str

    # AI Pricing Tier ("free" or "paid")
    AI_PRICING_MODE: str = "paid"

    AI_CURRENCY: str = "USD"


    # Gemini Text
    GEMINI_INPUT_PRICE_PER_1M: float = 1.50

    GEMINI_OUTPUT_PRICE_PER_1M: float = 7.50

    # Groq
    GROQ_INPUT_PRICE_PER_1M: float = 0.15

    GROQ_OUTPUT_PRICE_PER_1M: float = 0.60

    # Gemini Live Text
    GEMINI_LIVE_TEXT_INPUT_PRICE_PER_1M: float = 0.75

    GEMINI_LIVE_TEXT_OUTPUT_PRICE_PER_1M: float = 4.50

    # Gemini Live Audio
    GEMINI_LIVE_AUDIO_INPUT_PRICE_PER_1M: float = 3.00

    GEMINI_LIVE_AUDIO_OUTPUT_PRICE_PER_1M: float = 12.00

    # Gemini audio tokenization:
    # 25 tokens / second
    GEMINI_AUDIO_TOKENS_PER_SECOND: int = 25

    # MongoDB
    MONGODB_URI: str
    DATABASE_NAME: str

    # Upload
    UPLOAD_DIR: str
    MAX_FILE_SIZE: int

    # RAG Settings
    RAG_ENABLED: bool = True
    RAG_ALLOWED_DOMAINS: str = "webandcrafts.com,www.webandcrafts.com"
    RAG_EMBEDDING_MODEL: str = "gemini-embedding-001"
    RAG_EMBEDDING_DIMENSIONS: int = 768
    RAG_VECTOR_WEIGHT: float = 0.7
    RAG_KEYWORD_WEIGHT: float = 0.3
    RAG_MIN_RELEVANCE_SCORE: float = 0.65
    RAG_TOP_K_VECTOR: int = 20
    RAG_TOP_K_KEYWORD: int = 20
    RAG_TOP_K_FINAL: int = 5
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 100
    RAG_MAX_PAGES: int = 1000
    RAG_MAX_DEPTH: int = 5
    RAG_REQUEST_TIMEOUT: int = 15
    RAG_CRAWL_DELAY: float = 1.0
    RAG_CONCURRENCY: int = 5

    @property
    def allowed_domains_list(self) -> list[str]:
        """Return parsed allowed domain list."""
        return [
            domain.strip().lower()
            for domain in self.RAG_ALLOWED_DOMAINS.split(",")
            if domain.strip()
        ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()