"""Runtime configuration. Everything comes from environment variables (no literals in code paths)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 6000
    log_level: str = "INFO"
    job_service_url: str = ""  # e.g. http://job:5002 ; empty => job lookup disabled (tests / local files)
    redis_url: str = ""  # optional; empty => in-process LRU
    ocr_concurrency: int = 2
    ocr_dpi: int = 200
    ocr_langs: str = "eng+vie"
    max_upload_bytes: int = 5 * 1024 * 1024
    max_pages: int = 5
    rate_limit_per_min: int = 10
    cache_ttl_seconds: int = 24 * 3600
    job_cache_ttl_seconds: int = 600
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: str = "models"
    busy_timeout_seconds: float = 30.0
    jwt_secret: str = ""  # empty => dev auth (X-User-Id header, same convention as the .NET services)
    jwt_issuer: str = "job-platform"
    jwt_audience: str = "job-platform"
    job_http_timeout_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
