from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool

    # Gemini
    GEMINI_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str

    # MongoDB
    MONGODB_URI: str
    DATABASE_NAME: str

    # ChromaDB
    CHROMA_PATH: str

    # -------------------------
    # Upload
    # -------------------------
    UPLOAD_DIR: str
    MAX_FILE_SIZE: int

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()