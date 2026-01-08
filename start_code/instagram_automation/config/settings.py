"""Instagram Automation - Configuration Module"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Instagram Accounts
    instagram_account_1: str = os.getenv("INSTAGRAM_TEST_ACCOUNT_1", "")
    instagram_password_1: str = os.getenv("INSTAGRAM_TEST_PASSWORD_1", "")
    instagram_account_2: str = os.getenv("INSTAGRAM_TEST_ACCOUNT_2", "")
    instagram_password_2: str = os.getenv("INSTAGRAM_TEST_PASSWORD_2", "")

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

    # Midjourney Configuration
    midjourney_api_key: str = os.getenv("MIDJOURNEY_API_KEY", "")
    midjourney_base_url: str = os.getenv("MIDJOURNEY_BASE_URL", "https://api.imaginepro.ai/api/v1")

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://dbuser:dbpassword@localhost:5432/instagram_automation"
    )

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Cost Budget
    monthly_budget: float = float(os.getenv("MONTHLY_BUDGET", "100.0"))
    daily_budget: float = float(os.getenv("DAILY_BUDGET", "3.33"))

    # Rate Limiting
    max_likes_per_hour: int = int(os.getenv("MAX_LIKES_PER_HOUR", "20"))
    max_follows_per_hour: int = int(os.getenv("MAX_FOLLOWS_PER_HOUR", "10"))
    max_comments_per_hour: int = int(os.getenv("MAX_COMMENTS_PER_HOUR", "5"))
    max_posts_per_day: int = int(os.getenv("MAX_POSTS_PER_DAY", "2"))
    max_daily_interactions: int = int(os.getenv("MAX_DAILY_INTERACTIONS", "25"))

    # Content Configuration
    content_language: str = os.getenv("CONTENT_LANGUAGE", "en")
    default_content_type: str = os.getenv("DEFAULT_CONTENT_TYPE", "product_review")
    enable_image_cache: bool = os.getenv("ENABLE_IMAGE_CACHE", "true").lower() == "true"

    # Account Configuration
    account_1_primary_category: str = os.getenv("ACCOUNT_1_PRIMARY_CATEGORY", "phone_case")
    account_1_secondary_categories: str = os.getenv("ACCOUNT_1_SECONDARY_CATEGORIES", "phone_film")
    account_2_primary_category: str = os.getenv("ACCOUNT_2_PRIMARY_CATEGORY", "earbuds")
    account_2_secondary_categories: str = os.getenv("ACCOUNT_2_SECONDARY_CATEGORIES", "noise_cancelling_headphone")
    shared_categories: str = os.getenv("SHARED_CATEGORIES", "charging_cable,charger")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/automation.log")

    # Proxy
    use_proxy: bool = os.getenv("USE_PROXY", "false").lower() == "true"
    proxy_url: Optional[str] = os.getenv("PROXY_URL")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
