from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from app.config import get_settings
from app.db.session import get_client
from app.repos.repo_container import get_repos
from app.services.embeddings_service import get_embeddings_service
from app.services.retrieval.service import get_retrieval_service
from app.services.synthesis.service import get_synthesis_service


def clear_app_caches() -> None:
    get_settings.cache_clear()
    get_client.cache_clear()
    get_repos.cache_clear()
    get_embeddings_service.cache_clear()
    get_retrieval_service.cache_clear()
    get_synthesis_service.cache_clear()


@pytest.fixture(autouse=True)
def clear_singletons() -> None:
    clear_app_caches()
    yield
    clear_app_caches()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def configure_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., None]:
    def _configure(
        *,
        openai_key: str = "test-key",
        chunk_size: int = 40,
        chunk_overlap: int = 8,
        top_k: int = 3,
        enable_synthesis: bool = True,
        enable_hybrid_retrieval: bool = True,
        retrieval_candidate_k: int = 8,
        vector_weight: float = 0.4,
        lexical_weight: float = 0.6,
    ) -> None:
        chroma_dir = tmp_path / ".chroma"
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
        monkeypatch.setenv("OPENAI_MODEL", "test-chat-model")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")
        monkeypatch.setenv("ENABLE_SYNTHESIS", str(enable_synthesis).lower())
        monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))
        monkeypatch.setenv("CHUNK_OVERLAP", str(chunk_overlap))
        monkeypatch.setenv("TOP_K", str(top_k))
        monkeypatch.setenv("ENABLE_HYBRID_RETRIEVAL", str(enable_hybrid_retrieval).lower())
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", str(retrieval_candidate_k))
        monkeypatch.setenv("VECTOR_WEIGHT", str(vector_weight))
        monkeypatch.setenv("LEXICAL_WEIGHT", str(lexical_weight))
        monkeypatch.setenv("DATA_DIR", str(data_dir))
        monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))
        monkeypatch.setenv("COLLECTION_NAME", "test_documents")
        clear_app_caches()

    return _configure
