"""Ollama-compatible mock embedding server for offline Open WebUI tests.

Exposes ``POST /api/embed`` exactly like Ollama's embedding endpoint so it can
be plugged in as ``RAG_EMBEDDING_ENGINE=ollama`` + ``RAG_OLLAMA_BASE_URL``.

Embeddings are deterministic (seeded from a SHA-256 of the text) and L2
normalized, so the same chunk always yields the same vector. This makes
migration assertions (Chroma vector == Milvus vector) and cosine search
reproducible without any real model or GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List


def embed_text(text: str, dim: int = 384) -> List[float]:
    """Return a deterministic, L2-normalized ``dim``-dimensional vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_batch(texts: List[str], dim: int = 384) -> List[List[float]]:
    return [embed_text(t, dim) for t in texts]


class _Handler(BaseHTTPRequestHandler):
    dim: int = 384

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802  (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "invalid json"}, status=400)
            return

        if self.path.rstrip("/") in ("/api/embed", "/v1/embeddings"):
            texts = data.get("input", [])
            embeddings = embed_batch(texts, self.server.dim)  # type: ignore[attr-defined]
            self._send({"embeddings": embeddings, "model": data.get("model", "mock")})
        elif self.path.rstrip("/") == "/api/chat":
            self._send(
                {
                    "model": data.get("model", "mock-model"),
                    "message": {
                        "role": "assistant",
                        "content": "Mock response from the test LLM.",
                    },
                    "done": True,
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_GET(self):  # noqa: N802  (http.server API)
        if self.path.rstrip("/") == "/api/tags":
            self._send(
                {
                    "models": [
                        {"name": "mock-model:latest", "model": "mock-model:latest"},
                        {"name": "mock-embedding:latest", "model": "mock-embedding:latest"},
                    ]
                }
            )
        elif self.path.rstrip("/") == "/api/version":
            self._send({"version": "0.0.0-mock"})
        else:
            self._send({"error": "not found"}, status=404)

    def log_message(self, *args, **kwargs):  # silence default logging
        return


def run_server(host: str = "127.0.0.1", port: int = 0, dim: int = 384) -> ThreadingHTTPServer:
    """Start the mock server in a background thread. Returns the server object."""
    server = ThreadingHTTPServer((host, port), _Handler)
    server.dim = dim  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def free_port(host: str = "127.0.0.1") -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
