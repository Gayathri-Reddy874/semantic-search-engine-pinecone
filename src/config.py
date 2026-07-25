"""
Application configuration.

All runtime configuration is sourced from environment variables (via a
.env file in local development, or real environment variables in
production/CI). No secrets are ever hard-coded in source.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Pinecone ---
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index_name: str = Field(default="rag-semantic")
    pinecone_namespace: str = Field(default="default")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    # --- Embedding model (integrated inference) ---
    embedding_model: str = Field(default="multilingual-e5-large")

    # --- App behavior ---
    max_upload_size_mb: int = Field(default=5, ge=1, le=50)
    default_top_k: int = Field(default=3, ge=1, le=10)
    max_top_k: int = Field(default=10, ge=1, le=50)
    request_timeout_seconds: int = Field(default=30, ge=1)
    log_level: str = Field(default="INFO")

    @field_validator("pinecone_api_key")
    @classmethod
    def _api_key_not_placeholder(cls, v: str) -> str:
        if not v or v.strip() in {"", "your-api-key-here"}:
            raise ValueError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env "
                "and fill in a real key, or set it in your environment."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance (singleton per process)."""
    return Settings()
