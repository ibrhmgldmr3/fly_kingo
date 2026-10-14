from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ortam ayarları. Değerler .env dosyasından veya ortam değişkenlerinden okunur."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://flyhire:flyhire@localhost:5433/flyhire"
    data_dir: Path = Path("data")

    llm_model: str = "anthropic/claude-sonnet-5-5"
    embedding_model: str = "intfloat/multilingual-e5-base"

    gmail_credentials_path: Path = Path("private/gmail_credentials.json")
    gmail_token_path: Path = Path("private/gmail_token.json")

    @property
    def flywire_dir(self) -> Path:
        return self.data_dir / "flywire"


@lru_cache
def get_settings() -> Settings:
    return Settings()
