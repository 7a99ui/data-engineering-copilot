from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:3b"
    POSTGRES_USER: str = "copilot"
    POSTGRES_PASSWORD: str = "copilot123"
    POSTGRES_DB: str = "copilot_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    APP_HOST: str = "0.0.0.0"      # ← ajouter
    APP_PORT: int = 8000            # ← ajouter

    class Config:
        env_file = ".env"

settings = Settings()