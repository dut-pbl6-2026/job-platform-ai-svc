"""Runtime configuration. Everything comes from environment variables (no literals in code paths)."""
import re
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# env templates ship __SET_VIA_INFISICAL__; a copied file must never turn that public literal into a working secret.
_PLACEHOLDER_SECRET = re.compile(r"^__[A-Z][A-Z0-9_]*__$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 6000
    log_level: str = "INFO"
    job_service_url: str = ""  # e.g. http://job:5002 ; empty => job lookup disabled (tests / local files)
    redis_url: str = ""  # optional; empty => in-process LRU
    ocr_concurrency: int = 2
    ocr_dpi: int = 200
    ocr_langs: str = "eng+vie"
    ocr_timeout_seconds: float = 20.0  # per tesseract call; a hung process must not hold a scoring slot forever
    max_upload_bytes: int = 5 * 1024 * 1024
    max_pages: int = 5
    rate_limit_per_min: int = 10
    cache_ttl_seconds: int = 24 * 3600
    score_retention_seconds: int = 30 * 24 * 3600  # how long a user can reopen a score by its score_id
    job_cache_ttl_seconds: int = 600
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: str = "models"
    busy_timeout_seconds: float = 30.0
    # Production: JWT_METADATA_ADDRESS (auth-svc discovery URL) is required; tokens are RS256 verified against its JWKS.
    # X-User-Id is trusted only when JWT_METADATA_ADDRESS is unset AND DEV_AUTH_ENABLED=true AND the .NET environment is Development.
    jwt_metadata_address: str = ""
    dev_auth_enabled: bool = False
    environment: str = Field(
        "Production",
        validation_alias=AliasChoices("DOTNET_ENVIRONMENT", "ASPNETCORE_ENVIRONMENT", "environment"),
    )
    jwt_issuer: str = "job-platform"
    jwt_audience: str = "job-platform"
    job_http_timeout_seconds: float = 5.0
    search_service_url: str = ""  # e.g. http://search:5003 ; empty => chat job search is unavailable
    search_http_timeout_seconds: float = 5.0
    chat_rate_per_hour: int = 20  # per user, token bucket with a one-hour window
    chat_session_ttl_seconds: int = 24 * 3600  # sliding: refreshed on every message
    # Bearer that search-svc presents to /api/internal/* (embedding). Environment only, no default in code.
    # Required outside Development (>= 32 characters, enforced at startup). Empty in Development => those endpoints refuse every call.
    ai_internal_token_for_search: str = ""
    # Bearer for app-svc's background scoring: accepted ONLY by POST /api/ai/score-resume, as user "app-svc-scoring".
    # Environment only. Empty => no scoring caller. Production requires >= 32 characters. Exempt from the per-user cap.
    ai_scoring_service_token: str = ""
    # Hashing vectors are a degraded fallback. Only used when the ONNX model fails to load AND this is set to "hashing".
    embedder_fallback: Literal["", "hashing"] = ""
    # Hard cap on the whole request body (multipart overhead is added on top of max_upload_bytes).
    max_body_slack_bytes: int = 64 * 1024

    @field_validator("ai_internal_token_for_search", "ai_scoring_service_token")
    @classmethod
    def _placeholder_secret_is_unset(cls, value: str) -> str:
        """Treat an env-template placeholder (__SET_VIA_INFISICAL__) exactly like an empty value."""
        return "" if _PLACEHOLDER_SECRET.match(value.strip()) else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
