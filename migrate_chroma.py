"""Offline migration: copy every ChromaDB collection into a target vector store.

Reuses Open WebUI's own ``ChromaClient`` and the target backend client
(``MilvusClient`` or ``QdrantClient``) so collection naming (``open_webui_`` /
``open-webui_`` prefix, ``-`` -> ``_`` for Milvus), schema/dimension creation,
``process_metadata`` and the text-length clamp all match what the running
application expects. No Open WebUI server is required.

Usage (env vars are also honoured):

    # Migrate to a Milvus server.
    python migrate_chroma.py --dest milvus \
        --chroma-path /path/to/data/vector_db \
        --milvus-uri http://127.0.0.1:19530 --verify

    # Migrate to a Qdrant server.
    python migrate_chroma.py --dest qdrant \
        --chroma-path /path/to/data/vector_db \
        --qdrant-uri http://127.0.0.1:6333 --verify

Run ``--dry-run`` first to list collections and counts without writing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from typing import List, Optional

# --- configure env BEFORE importing open_webui (config reads env at import) ---
def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dest", choices=["milvus", "qdrant"], default=os.getenv("MIGRATE_DEST", "milvus"))
    p.add_argument("--chroma-path", default=os.getenv("CHROMA_DATA_PATH"))

    # Milvus target
    p.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI"))
    p.add_argument("--milvus-db", default=os.getenv("MILVUS_DB"))
    p.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    p.add_argument("--metric", default=os.getenv("MILVUS_METRIC_TYPE", "COSINE"))
    p.add_argument("--index-type", default=os.getenv("MILVUS_INDEX_TYPE", "HNSW"))
    p.add_argument("--multitenancy", default=os.getenv("ENABLE_MILVUS_MULTITENANCY_MODE", "false"))

    # Qdrant target
    p.add_argument("--qdrant-uri", default=os.getenv("QDRANT_URI"))
    p.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    p.add_argument("--qdrant-prefix", default=os.getenv("QDRANT_COLLECTION_PREFIX", "open-webui"))
    p.add_argument("--qdrant-on-disk", default=os.getenv("QDRANT_ON_DISK", "false"))
    p.add_argument("--qdrant-prefer-grpc", default=os.getenv("QDRANT_PREFER_GRPC", "false"))


def _apply_env(a: argparse.Namespace) -> None:
    """Push parsed CLI flags into the environment so open_webui.config picks them up."""
    if getattr(a, "chroma_path", None):
        os.environ["CHROMA_DATA_PATH"] = a.chroma_path

    dest = getattr(a, "dest", "milvus")
    if dest == "milvus":
        if getattr(a, "milvus_uri", None):
            os.environ["MILVUS_URI"] = a.milvus_uri
        if getattr(a, "milvus_db", None) is not None:
            os.environ["MILVUS_DB"] = a.milvus_db
        if getattr(a, "milvus_token", None) is not None:
            os.environ["MILVUS_TOKEN"] = a.milvus_token
        os.environ["MILVUS_METRIC_TYPE"] = getattr(a, "metric", "COSINE")
        os.environ["MILVUS_INDEX_TYPE"] = getattr(a, "index_type", "HNSW")
        os.environ["ENABLE_MILVUS_MULTITENANCY_MODE"] = str(getattr(a, "multitenancy", "false")).lower()
    elif dest == "qdrant":
        if getattr(a, "qdrant_uri", None):
            os.environ["QDRANT_URI"] = a.qdrant_uri
        if getattr(a, "qdrant_api_key", None) is not None:
            os.environ["QDRANT_API_KEY"] = a.qdrant_api_key
        os.environ["QDRANT_COLLECTION_PREFIX"] = getattr(a, "qdrant_prefix", "open-webui")
        os.environ["QDRANT_ON_DISK"] = str(getattr(a, "qdrant_on_disk", "false")).lower()
        os.environ["QDRANT_PREFER_GRPC"] = str(getattr(a, "qdrant_prefer_grpc", "false")).lower()


# Pre-parse a minimal set of flags purely to set env before open_webui import.
_pre = argparse.ArgumentParser(add_help=False)
_add_args(_pre)
_apply_env(_pre.parse_known_args()[0])

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("migrate_chroma")

# open_webui is imported from the sibling Open WebUI checkout (it is not a
# pip-installable package). Resolve it via OPEN_WEBUI_BACKEND, falling back to
# the conventional relative location used in this workspace.
_BACKEND_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "open-webui-0.10.2", "backend")
)
BACKEND = os.environ.get("OPEN_WEBUI_BACKEND") or _BACKEND_DEFAULT
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.retrieval.vector.dbs.chroma import ChromaClient  # noqa: E402


def make_dst_client(dest: str):
    """Instantiate the target backend client (reads open_webui.config, already set)."""
    if dest == "milvus":
        from open_webui.retrieval.vector.dbs.milvus import MilvusClient

        return MilvusClient()
    if dest == "qdrant":
        from open_webui.retrieval.vector.dbs.qdrant import QdrantClient

        return QdrantClient()
    raise ValueError(f"unknown destination backend: {dest}")


def _backend_of(client) -> str:
    mod = type(client).__module__
    if mod.endswith(".qdrant") or mod.endswith(".qdrant_multitenancy"):
        return "qdrant"
    if mod.endswith(".milvus") or mod.endswith(".milvus_multitenancy"):
        return "milvus"
    return "unknown"


def migrate(
    src_client: ChromaClient,
    dst_client,
    batch_size: int = 500,
    collection_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Read every Chroma collection (with embeddings) and upsert into the target.

    Returns a summary dict: ``{"dest": str, "collections": [(name, n), ...], "total": int}``.
    """
    chroma = src_client.client
    collections = chroma.list_collections()
    names = [c.name if hasattr(c, "name") else c for c in collections]
    log.info(f"Found {len(names)} Chroma collection(s): {names}")

    summary: dict = {"dest": _backend_of(dst_client), "collections": [], "total": 0}
    for name in names:
        if collection_filter and name not in collection_filter:
            log.info(f"[skip] {name}: not in --collections filter")
            continue

        coll = chroma.get_collection(name=name)
        # Include embeddings! Open WebUI's wrapped .get() omits them.
        result = coll.get(include=["embeddings", "documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        embeds = result.get("embeddings")
        if embeds is None:
            embeds = []
        metas = result.get("metadatas") or []

        if not ids:
            log.info(f"[skip] {name}: empty")
            summary["collections"].append((name, 0))
            continue

        items = [
            {
                "id": i,
                "text": (docs[idx] or ""),
                "vector": embeds[idx],
                "metadata": metas[idx] or {},
            }
            for idx, i in enumerate(ids)
        ]

        if dry_run:
            log.info(f"[dry-run] {name}: would migrate {len(items)} vector(s)")
            summary["collections"].append((name, len(items)))
            summary["total"] += len(items)
            continue

        for start in range(0, len(items), batch_size):
            dst_client.upsert(
                collection_name=name, items=items[start : start + batch_size]
            )
        log.info(f"[done] {name}: migrated {len(items)} vector(s)")
        summary["collections"].append((name, len(items)))
        summary["total"] += len(items)

    return summary


def _milvus_all_ids(client, collection_name: str) -> set:
    """Return the set of all ids in a Milvus collection.

    Uses the high-level ``query`` (which reads flushed data, no top-k limit)
    rather than ``get``/``search``: ``search`` requires a flush before
    freshly-upserted data becomes visible, so we explicitly ``flush`` first and
    paginate with ``limit``/``offset`` (Milvus caps those at 16384).
    """
    full = f"{client.collection_prefix}_{collection_name.replace('-', '_')}"
    try:
        client.client.flush(collection_name=full)
    except Exception:
        pass
    ids: set = set()
    offset = 0
    while True:
        res = client.client.query(
            collection_name=full, filter="", output_fields=["id"], limit=16384, offset=offset
        )
        if not res:
            break
        ids.update(r.get("id") for r in res)
        if len(res) < 16384:
            break
        offset += 16384
    return ids


def _milvus_all_docs(client, collection_name: str) -> dict:
    """Return ``{id: text}`` for every entity, via ``query`` (no index needed)."""
    full = f"{client.collection_prefix}_{collection_name.replace('-', '_')}"
    try:
        client.client.flush(collection_name=full)
    except Exception:
        pass
    out: dict = {}
    offset = 0
    while True:
        res = client.client.query(
            collection_name=full,
            filter="",
            output_fields=["id", "data"],
            limit=16384,
            offset=offset,
        )
        if not res:
            break
        for r in res:
            data = r.get("data") or {}
            out[r.get("id")] = data.get("text") if isinstance(data, dict) else None
        if len(res) < 16384:
            break
        offset += 16384
    return out


def _qdrant_all_ids(client, collection_name: str) -> set:
    """Return the set of all point ids in a Qdrant collection (scroll + offset).

    Qdrant returns point ids as ``uuid.UUID`` (for REST, as a hyphenated
    string). Normalize to the compact 32-char hex to match Chroma's ids.
    """
    full = f"{client.collection_prefix}_{collection_name}"
    ids: set = set()
    offset = None
    while True:
        points, offset = client.client.scroll(collection_name=full, limit=16384, offset=offset)
        for p in points:
            ids.add(_norm_qdrant_id(p.id))
        if offset is None:
            break
    return ids


def _qdrant_all_docs(client, collection_name: str) -> dict:
    """Return ``{id: text}`` for every point, via ``scroll`` (no index needed)."""
    full = f"{client.collection_prefix}_{collection_name}"
    out: dict = {}
    offset = None
    while True:
        points, offset = client.client.scroll(collection_name=full, limit=16384, offset=offset)
        for p in points:
            out[_norm_qdrant_id(p.id)] = (p.payload or {}).get("text")
        if offset is None:
            break
    return out


def _norm_qdrant_id(pid) -> str:
    """Normalize a Qdrant point id to a compact 32-char hex string.

    Qdrant stores UUID-style ids; the REST API returns them as hyphenated
    strings (or ``uuid.UUID`` objects), while Chroma stores the compact hex.
    Falls back to ``str`` for non-UUID (e.g. integer) ids.
    """
    if isinstance(pid, uuid.UUID):
        return pid.hex
    if isinstance(pid, str):
        try:
            return uuid.UUID(pid).hex
        except ValueError:
            return pid
    return str(pid)


def _all_ids(client, collection_name: str) -> set:
    if _backend_of(client) == "qdrant":
        return _qdrant_all_ids(client, collection_name)
    return _milvus_all_ids(client, collection_name)


def _all_docs(client, collection_name: str) -> dict:
    if _backend_of(client) == "qdrant":
        return _qdrant_all_docs(client, collection_name)
    return _milvus_all_docs(client, collection_name)


def verify(src_client: ChromaClient, dst_client) -> bool:
    """Compare per-collection id sets between Chroma and the target store."""
    chroma = src_client.client
    ok = True
    for c in chroma.list_collections():
        name = c.name if hasattr(c, "name") else c
        chroma_ids = set(chroma.get_collection(name).get(include=[])["ids"])
        dst_ids = _all_ids(dst_client, name)
        match = chroma_ids == dst_ids
        ok = ok and match
        log.info(
            f"[verify] {name}: chroma={len(chroma_ids)} dst={len(dst_ids)} "
            f"{'OK' if match else 'MISMATCH'}"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate ChromaDB -> target vector store (offline)")
    _add_args(parser)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Only migrate these specific collection names.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    # Re-apply env from the *full* parser so CLI overrides win.
    _apply_env(args)

    src = ChromaClient()
    dst = make_dst_client(args.dest)

    summary = migrate(
        src,
        dst,
        batch_size=args.batch,
        collection_filter=args.collections,
        dry_run=args.dry_run,
    )
    log.info(
        f"Summary ({summary['dest']}): {summary['total']} vectors across "
        f"{len(summary['collections'])} collection(s)"
    )

    if args.verify and not args.dry_run:
        if not verify(src, dst):
            log.error("Verification FAILED: collection counts differ.")
            return 1
        log.info("Verification PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
