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

    # MongoDB
    MONGODB_URI: str
    DATABASE_NAME: str

    # Upload
    UPLOAD_DIR: str
    MAX_FILE_SIZE: int

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()