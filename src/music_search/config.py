"""Application settings loaded from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared by ingestion, retrieval, and migrations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MUSIC_SEARCH_",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "postgresql+psycopg://music_search:music_search@localhost:5432/music_search"
    clap_model_name: str = "laion/clap-htsat-unfused"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    sample_rate: int = Field(default=48_000, ge=48_000, le=48_000)
    window_seconds: float = Field(default=10.0, gt=0, le=10.0)
    stride_seconds: float = Field(default=5.0, gt=0)
    embedding_dimension: int = Field(default=512, ge=512, le=512)
    embedding_batch_size: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def validate_audio_window(self) -> "Settings":
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride_seconds must be less than or equal to window_seconds")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance per process."""

    return Settings()
