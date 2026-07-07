"""End-to-end tests: run the full Open WebUI server, create knowledge bases
through its API, migrate the vector store, and verify knowledge is still usable
after switching to the new backend.

These tests spin up a real Open WebUI backend (uvicorn subprocess) backed by
mock embedding/chat servers.  They exercise the **exact** code paths a user
would hit: file upload, KB creation, RAG query -- then Chroma -> target
migration via the CLI, and finally query + list after switching.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest
import requests

from tests.mock_embedding_server import free_port

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _start_openwebui(data_dir: str, vector_db: str, ollama_url: str, port: int, **extra_env):
    """Launch the full Open WebUI server in a uvicorn subprocess."""
    env = {
        **os.environ,
        "WEBUI_AUTH": "false",
        "WEBUI_SECRET_KEY": "test-secret-e2e",
        "DATA_DIR": data_dir,
        "VECTOR_DB": vector_db,
        "CHROMA_DATA_PATH": os.path.join(data_dir, "vector_db"),
        "RAG_EMBEDDING_ENGINE": "ollama",
        "RAG_EMBEDDING_MODEL": "mock-embedding",
        "RAG_OLLAMA_BASE_URL": ollama_url,
        "OLLAMA_BASE_URL": ollama_url,
        "BYPASS_MODEL_ACCESS_CONTROL": "true",
        "OFFLINE_MODE": "true",
        "ENABLE_RAG_WEB_SEARCH": "false",
        "SAFE_MODE": "false",
        **extra_env,
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "open_webui.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/health", timeout=90)
    except Exception:
        proc.kill(); proc.wait()
        raise
    return proc, f"http://127.0.0.1:{port}"


def _wait_http(url: str, timeout: int = 90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"HTTP service at {url} did not become healthy")


def _signin(base_url: str) -> str:
    r = requests.post(
        f"{base_url}/api/v1/auths/signin",
        json={"email": "admin@localhost", "password": "admin"},
    )
    assert r.status_code == 200, f"signin failed [{r.status_code}]: {r.text}"
    return r.json()["token"]


def _api_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _upload_and_index(base_url: str, kb_id: str, content: str, token: str) -> str:
    files = {"file": ("test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    data = {"metadata": json.dumps({"knowledge_id": kb_id})}
    r = requests.post(
        f"{base_url}/api/v1/files/",
        files=files, data=data, headers=_api_headers(token),
    )
    assert r.status_code == 200, f"file upload failed [{r.status_code}]: {r.text}"
    return r.json()["id"]


def _wait_chroma_populated(chroma_data_path: str, kb_ids: list[str], timeout: int = 30) -> dict:
    """Poll Chroma until every KB collection has data.  Returns ``{name: count}``."""
    import open_webui.retrieval.vector.dbs.chroma as chroma_mod
    from open_webui.retrieval.vector.dbs.chroma import ChromaClient

    orig = chroma_mod.CHROMA_DATA_PATH
    chroma_mod.CHROMA_DATA_PATH = chroma_data_path
    try:
        cc = ChromaClient()
        deadline = time.time() + timeout
        counts = {}
        while time.time() < deadline:
            all_ok = True
            for kb_id in kb_ids:
                try:
                    cnt = cc.client.get_collection(kb_id).count()
                    counts[kb_id] = cnt
                    if cnt == 0:
                        all_ok = False
                except Exception:
                    all_ok = False
                    break
            if all_ok:
                return counts
            time.sleep(1)
        raise RuntimeError(f"Chroma not populated for {kb_ids}: {counts}")
    finally:
        chroma_mod.CHROMA_DATA_PATH = orig


def _run_migration(data_dir: str, dest: str, env_extra: dict) -> int:
    migrate_script = os.path.join(PROJECT_ROOT, "migrate_chroma.py")
    env = {
        **os.environ,
        "CHROMA_DATA_PATH": os.path.join(data_dir, "vector_db"),
        "WEBUI_SECRET_KEY": "test",
        "OPEN_WEBUI_BACKEND": os.environ.get(
            "OPEN_WEBUI_BACKEND",
            os.path.join(PROJECT_ROOT, "..", "open-webui-0.10.2", "backend"),
        ),
        **env_extra,
    }
    result = subprocess.run(
        [sys.executable, migrate_script, "--dest", dest, "--verify"],
        env=env, capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT,
    )
    print("--- migrate stdout ---")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    print("--- migrate stderr ---")
    print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    return result.returncode


@pytest.mark.parametrize("target", ["qdrant"])
def test_e2e_chroma_migration_and_query(
    target, qdrant_server, mock_embedding_server, monkeypatch
):
    data_dir = tempfile.mkdtemp(prefix="owui_e2e_")
    try:
        shutil.rmtree(data_dir, ignore_errors=True)
        os.makedirs(data_dir, exist_ok=True)

        port = free_port()
        mock_url = mock_embedding_server

        # ── Phase 1: OpenWebUI with Chroma ─────────────────────────────────
        proc1, base_url = _start_openwebui(data_dir, "chroma", mock_url, port)
        try:
            token = _signin(base_url)
            h = _api_headers(token)

            kb_ids = []
            for suffix in ("Alpha", "Beta"):
                r = requests.post(
                    f"{base_url}/api/v1/knowledge/create",
                    json={"name": f"e2e-kb-{suffix}", "description": f"E2E {suffix}"},
                    headers=h,
                )
                assert r.status_code == 200, f"kb create failed: {r.text}"
                kb_ids.append(r.json()["id"])

            for kb_id in kb_ids:
                content = f"E2E_UNIQUE_{uuid.uuid4().hex[:8]}_e2e_test_sentence"
                _upload_and_index(base_url, kb_id, content, token)

            # Poll Chroma directly until every KB has stored vectors.
            chroma_counts = _wait_chroma_populated(
                os.path.join(data_dir, "vector_db"), kb_ids
            )
            for kb_id in kb_ids:
                assert chroma_counts.get(kb_id, 0) > 0, f"KB {kb_id[:8]} has no Chroma data"

        finally:
            proc1.terminate()
            proc1.wait(timeout=30)

        # ── Phase 2: offline migration via CLI ─────────────────────────────
        if target == "qdrant":
            dest_env = {
                "QDRANT_URI": os.environ["QDRANT_URI"],
                "QDRANT_COLLECTION_PREFIX": "open-webui",
                "QDRANT_ON_DISK": "false",
                "QDRANT_PREFER_GRPC": "false",
            }
            dest = "qdrant"
        else:
            dest_env = {
                "MILVUS_URI": os.environ["MILVUS_URI"],
                "MILVUS_DB": os.environ.get("MILVUS_DB", "default"),
                "MILVUS_METRIC_TYPE": "COSINE",
                "ENABLE_MILVUS_MULTITENANCY_MODE": "false",
            }
            dest = "milvus"

        assert _run_migration(data_dir, dest, dest_env) == 0

        # ── Phase 3: restart with new backend ──────────────────────────────
        port2 = free_port()
        proc2, base_url2 = _start_openwebui(
            data_dir, dest, mock_url, port2, **dest_env
        )
        try:
            token2 = _signin(base_url2)
            h2 = _api_headers(token2)

            # KBs must still be listed.
            r = requests.get(f"{base_url2}/api/v1/knowledge/", headers=h2)
            assert r.status_code == 200
            kb_items = (r.json() or {}).get("items") or []
            for kb_id in kb_ids:
                assert any(k["id"] == kb_id for k in kb_items), (
                    f"KB {kb_id[:8]} missing post-migration"
                )

            # Retrieval endpoint must be alive (search/migration fidelity is
            # tested exhaustively in test_migrated_store_works_with_openwebui).
            for kb_id in kb_ids:
                r = requests.post(
                    f"{base_url2}/api/v1/retrieval/query/doc",
                    json={"collection_name": kb_id, "query": "E2E", "k": 3},
                    headers=h2,
                )
                assert r.status_code == 200, f"retrieval endpoint failed: {r.text}"

        finally:
            proc2.terminate()
            proc2.wait(timeout=30)

    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
