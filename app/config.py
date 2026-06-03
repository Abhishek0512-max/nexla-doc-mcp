"""Central application configuration.

Loaded once from the environment / .env file. Every other module imports
`get_settings()` rather than reading os.environ directly, so there is exactly
one typed, validated source of truth.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # --- OpenAI (embeddings + synthesis share one key) ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"  # any chat model on your account
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Behaviour ---
    enable_synthesis: bool = True  # hybrid switch: synthesize prose vs. return ranked chunks

    # --- Chunking + retrieval ---
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    enable_hybrid_retrieval: bool = True
    retrieval_candidate_k: int = Field(default=80, ge=1)
    vector_weight: float = Field(default=0.4, ge=0)
    lexical_weight: float = Field(default=0.6, ge=0)

    # --- Storage ---
    data_dir: Path = Path("data")
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "documents"

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.vector_weight + self.lexical_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    """Build settings once, then reuse the cached instance process-wide."""
    return Settings()
