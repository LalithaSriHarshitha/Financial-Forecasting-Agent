"""Application settings, loaded from environment variables / the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Gemini
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    # Free-tier friendly limits; raise them if your key is on a paid tier.
    llm_requests_per_minute: int = Field(default=8, alias="LLM_REQUESTS_PER_MINUTE")
    embed_requests_per_minute: int = Field(default=90, alias="EMBED_REQUESTS_PER_MINUTE")

    # MySQL
    mysql_host: str = Field(default="localhost", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="forecast_user", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="tcs_forecast", alias="MYSQL_DATABASE")
    # Optional full SQLAlchemy URL; overrides the MYSQL_* values (used by tests).
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")

    # Paths
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def database_url(self) -> str | URL:
        if self.database_url_override:
            return self.database_url_override
        # URL.create escapes special characters (e.g. '@') in the password.
        return URL.create(
            "mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )

    def _abs(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_path(self) -> Path:
        return self._abs(self.data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_path / "processed"

    @property
    def manifest_path(self) -> Path:
        return self.data_path / "manifest.json"

    @property
    def chroma_path(self) -> Path:
        return self._abs(self.chroma_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
