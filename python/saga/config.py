from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def windows_user_environment(name: str) -> str:
    """Read a newly-set Windows user variable even when PyCharm predates the change."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except (ImportError, FileNotFoundError, OSError):
        return ""


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_hub_api_key: str = Field(
        default_factory=lambda: windows_user_environment("OPEN_AI_SERVICE_HUB_API_KEY"),
        alias="OPEN_AI_SERVICE_HUB_API_KEY",
    )
    service_hub_base_url: str = Field(
        default="https://open.hasa.re.kr/v1", alias="SAGA_SERVICE_HUB_BASE_URL"
    )
    service_hub_model: str = Field(default="gpt-oss-20b", alias="SAGA_MODEL")
    service_hub_fast_model: str = Field(default="gpt-oss-20b", alias="SAGA_FAST_MODEL")
    service_hub_direct_model: str = Field(default="llama-3.3-70b", alias="SAGA_DIRECT_MODEL")
    groq_api_key: str = Field(
        default_factory=lambda: windows_user_environment("GROQ_API_KEY"), alias="GROQ_API_KEY"
    )
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    groq_fast_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_FAST_MODEL")
    groq_direct_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_DIRECT_MODEL")

    @model_validator(mode="after")
    def use_current_windows_groq_key(self) -> "Settings":
        # PyCharm may keep an old process environment after a Windows user
        # variable is updated. Read the current user value at server startup.
        current_user_key = windows_user_environment("GROQ_API_KEY").strip()
        if current_user_key:
            self.groq_api_key = current_user_key
        return self
    service_hub_vision_model: str = Field(default="qwen2.5-vl-72b", alias="SAGA_VISION_MODEL")
    reasoning_effort: str = Field(default="low", alias="SAGA_REASONING_EFFORT")
    host: str = Field(default="127.0.0.1", alias="SAGA_HOST")
    port: int = Field(default=8090, alias="SAGA_PORT")
    upload_dir: Path = Field(default=PROJECT_ROOT / "saga-uploads", alias="SAGA_UPLOAD_DIR")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="SAGA_DATA_DIR")
    database_path: Path = Field(default=PROJECT_ROOT / "data" / "saga.db", alias="SAGA_DATABASE_PATH")
    retrieval_limit: int = Field(default=32, ge=5, le=100, alias="SAGA_RETRIEVAL_LIMIT")
    context_limit: int = Field(default=24, ge=3, le=30, alias="SAGA_CONTEXT_LIMIT")
    max_context_chars: int = Field(default=42000, ge=5000, le=100000, alias="SAGA_MAX_CONTEXT_CHARS")
    hybrid_search_enabled: bool = Field(default=True, alias="SAGA_HYBRID_SEARCH_ENABLED")
    vector_model: str = Field(default="hash-ngram-v1", alias="SAGA_VECTOR_MODEL")
    answer_review_enabled: bool = Field(default=True, alias="SAGA_ANSWER_REVIEW_ENABLED")
    answer_review_model: str = Field(default="", alias="SAGA_ANSWER_REVIEW_MODEL")
    answer_length: Literal["concise", "standard", "detailed", "very_detailed"] = Field(
        default="standard", alias="SAGA_ANSWER_LENGTH"
    )
    # National Law Information Center Open API.  OC is a separate service
    # credential from the LLM key and is intentionally empty by default.
    law_api_oc: str = Field(
        default_factory=lambda: windows_user_environment("SAGA_LAW_API_OC"),
        alias="SAGA_LAW_API_OC",
    )
    law_api_base_url: str = Field(
        default="https://www.law.go.kr/DRF", alias="SAGA_LAW_API_BASE_URL"
    )
    law_api_timeout: float = Field(default=30.0, ge=5.0, le=120.0, alias="SAGA_LAW_API_TIMEOUT")
    law_pdf_dir: Path = Field(
        default=PROJECT_ROOT / "saga-uploads", alias="SAGA_LAW_PDF_DIR"
    )
    external_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "external", alias="SAGA_EXTERNAL_DATA_DIR"
    )
    admin_token: str = Field(default="", alias="SAGA_ADMIN_TOKEN")

    @field_validator(
        "upload_dir", "data_dir", "database_path", "law_pdf_dir", "external_data_dir", mode="before"
    )
    @classmethod
    def resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.law_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.external_data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
