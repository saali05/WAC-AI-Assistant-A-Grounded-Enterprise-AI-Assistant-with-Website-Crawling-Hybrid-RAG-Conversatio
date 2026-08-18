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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()