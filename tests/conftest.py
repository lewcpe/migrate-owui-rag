"""Pytest fixtures for offline Chroma -> Milvus migration tests.

These fixtures spin up a mock Ollama embedding server and provide isolated
Chroma / Milvus clients plus an ``ingest_documents`` helper that faithfully
replays Open WebUI's ``save_docs_to_vector_db`` pipeline (chunk -> embed ->
insert) without needing the full web server.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Ensure the backend package is importable regardless of pytest's rootdir.
# open_webui is imported from the sibling Open WebUI checkout (not pip-installable);
# resolve via OPEN_WEBUI_BACKEND, falling back to the conventional relative path.
_BACKEND_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "open-webui-0.10.2", "backend")
)
BACKEND = os.environ.get("OPEN_WEBUI_BACKEND") or _BACKEND_DEFAULT
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Project root must be on sys.path so `import migrate_chroma_to_milvus` and
# `from tests...` resolve under pytest's default import mode.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure Open WebUI *before* any open_webui module is imported.
os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key-for-migration-tests")
os.environ["VECTOR_DB"] = "chroma"
os.environ["RAG_EMBEDDING_ENGINE"] = "ollama"
os.environ["RAG_EMBEDDING_MODEL"] = "mock-embedding"
os.environ["RAG_EMBEDDING_BATCH_SIZE"] = "32"
os.environ["RAG_TEXT_SPLITTER"] = "character"
os.environ["CHUNK_SIZE"] = "1000"
os.environ["CHUNK_OVERLAP"] = "100"
os.environ["ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER"] = "False"
os.environ["MILVUS_METRIC_TYPE"] = "COSINE"
os.environ["MILVUS_INDEX_TYPE"] = "HNSW"
os.environ["ENABLE_MILVUS_MULTITENANCY_MODE"] = "false"
os.environ["OFFLINE_MODE"] = "true"

from open_webui.config import (  # noqa: E402
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_MODEL,
)
from open_webui.retrieval.utils import get_embedding_function  # noqa: E402
from open_webui.retrieval.vector.dbs.chroma import ChromaClient  # noqa: E402
from open_webui.retrieval.vector.dbs.milvus import MilvusClient  # noqa: E402
from open_webui.utils.misc import sanitize_text_for_db  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from tests.mock_embedding_server import run_server  # noqa: E402

MOCK_DIM = 384


@pytest.fixture(scope="session")
def mock_embedding_server():
    port = __import__("tests.mock_embedding_server", fromlist=["free_port"]).free_port()
    server = run_server(port=port, dim=MOCK_DIM)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    # Keep RAG_OLLAMA_BASE_URL pointed at the mock for the whole session.
    os.environ["RAG_OLLAMA_BASE_URL"] = base_url
    yield base_url
    server.shutdown()


@pytest.fixture(scope="session")
def embedding_function(mock_embedding_server):
    """Async embedding function wired to the mock Ollama server."""
    return get_embedding_function(
        embedding_engine=RAG_EMBEDDING_ENGINE,
        embedding_model=RAG_EMBEDDING_MODEL,
        embedding_function=None,
        url=mock_embedding_server,
        key="",
        embedding_batch_size=int(os.environ["RAG_EMBEDDING_BATCH_SIZE"]),
        enable_async=False,
    )


def _wait_for_port(host: str, port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


@pytest.fixture(scope="session")
def milvus_server():
    """Start a real Milvus server via docker compose (session-scoped).

    Uses ``docker-compose.milvus.yml`` (etcd + rustfs + milvus). Skipped
    automatically if Docker / docker compose is unavailable. The migration
    target is a Milvus *server* (not Milvus Lite), so tests exercise the same
    code path used in production.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not available")

    compose_bin = "docker" if shutil.which("docker") and _compose_present() else None
    if compose_bin is None:
        pytest.skip("docker compose not available")

    compose_file = str(Path(__file__).resolve().parent.parent / "docker-compose.milvus.yml")
    host, port = "127.0.0.1", 19530

    try:
        subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d", "--wait"],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"could not start Milvus stack: {e}")

    if not _wait_for_port(host, port, timeout=240):
        subprocess.run(["docker", "compose", "-f", compose_file, "down", "-v"], capture_output=True)
        pytest.skip("Milvus server did not become ready in time")

    os.environ["MILVUS_URI"] = f"http://{host}:{port}"
    os.environ["MILVUS_DB"] = "default"

    # Wait until the proxy actually accepts requests (not just TCP open).
    ready = False
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            MilvusClient().client.list_collections()
            ready = True
            break
        except Exception:
            time.sleep(2)
    if not ready:
        subprocess.run(["docker", "compose", "-f", compose_file, "down", "-v"], capture_output=True)
        pytest.skip("Milvus server did not become ready (proxy) in time")

    yield f"http://{host}:{port}"

    subprocess.run(["docker", "compose", "-f", compose_file, "down", "-v"], capture_output=True)


def _compose_present() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "compose", "version"], capture_output=True, timeout=20
            ).returncode
            == 0
        )
    except Exception:
        return False


@pytest.fixture
def chroma_client(tmp_path, monkeypatch):
    data = tmp_path / "vector_db"
    monkeypatch.setattr("open_webui.config.CHROMA_DATA_PATH", str(data))
    return ChromaClient()


@pytest.fixture
def milvus_client(milvus_server, monkeypatch):
    # Point at the Docker Milvus server (set by the milvus_server fixture).
    monkeypatch.setattr("open_webui.config.MILVUS_URI", os.environ["MILVUS_URI"])
    monkeypatch.setattr("open_webui.config.MILVUS_DB", os.environ.get("MILVUS_DB", "default"))
    monkeypatch.setattr("open_webui.config.ENABLE_MILVUS_MULTITENANCY_MODE", False)
    client = MilvusClient()
    client.reset()  # isolate each test: drop any open_webui_* collections
    return client


def ingest_documents(
    client,
    collection_name: str,
    raw_docs: list[dict],
    embedding_function,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    """Emulate ``save_docs_to_vector_db``: chunk -> embed -> insert.

    ``raw_docs`` is a list of ``{"title": str, "content": str, "file_id": str,
    "hash": str}``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    docs = [
        Document(
            page_content=d["content"],
            metadata={
                "name": d.get("title", ""),
                "source": d.get("title", ""),
                "file_id": d.get("file_id", ""),
                "hash": d.get("hash", ""),
            },
        )
        for d in raw_docs
    ]
    split_docs = splitter.split_documents(docs)
    if not split_docs:
        raise ValueError("no chunks produced")

    texts = [sanitize_text_for_db(doc.page_content) for doc in split_docs]
    metadatas = [
        {
            **doc.metadata,
            "embedding_config": {
                "engine": RAG_EMBEDDING_ENGINE,
                "model": RAG_EMBEDDING_MODEL,
            },
        }
        for doc in split_docs
    ]

    embeddings = asyncio.run(
        embedding_function(
            list(map(lambda x: x.replace("\n", " "), texts)),
            prefix=RAG_EMBEDDING_CONTENT_PREFIX,
        )
    )

    items = [
        {
            "id": __import__("uuid").uuid4().hex,
            "text": text,
            "vector": embeddings[idx],
            "metadata": metadatas[idx],
        }
        for idx, text in enumerate(texts)
    ]
    client.insert(collection_name=collection_name, items=items)
    return len(items)
