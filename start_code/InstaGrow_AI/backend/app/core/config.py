from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    APP_NAME: str = "InstaGrow AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-here"

    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/instagrow_ai"

    JWT_SECRET_KEY: str = "your-jwt-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 10080

    OPENAI_API_KEY: str = ""
    APIFY_API_TOKEN: str = ""

    CORS_ORIGINS: List[str] = ["http://localhost:3000"]


settings = Settings()
