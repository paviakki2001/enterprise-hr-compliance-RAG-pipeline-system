from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    BACKEND_BASE_URL: str = "http://localhost:8000"
    BACKEND_QUERY_PATH: str = "/query"
    HEALTH_PATH: str = "/health"
    REQUEST_TIMEOUT_SECONDS: int = 90
    APP_TITLE: str = "HR Compliance RAG Assistant"
    LOG_LEVEL: str = "INFO"

    DEFAULT_TOP_K_RETRIEVAL: int = 8
    DEFAULT_TOP_K_CONTEXT: int = 4
    SHOW_CITATIONS_DEFAULT: bool = True

settings = Settings()
