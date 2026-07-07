"""Offline migration: copy every ChromaDB collection into Milvus.

Reuses Open WebUI's own ``ChromaClient`` / ``MilvusClient`` so collection
naming (``open_webui_`` prefix + ``-`` -> ``_``), schema/dimension creation,
``process_metadata`` and the text-length clamp all match what the running
application expects. No Open WebUI server is required.

Usage (env vars are also honoured, see MIGRATE_VECTOR.md):

    python migrate_chroma_to_milvus.py \
        --chroma-path /path/to/data/vector_db \
        --milvus-uri  /path/to/data/vector_db/milvus.db \
        --batch 500 \
        --verify

Run ``--dry-run`` first to list collections and counts without writing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

# --- configure env BEFORE importing open_webui (config reads env at import) ---
def _apply_cli_env() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--chroma-path", default=os.getenv("CHROMA_DATA_PATH"))
    p.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI"))
    p.add_argument("--milvus-db", default=os.getenv("MILVUS_DB"))
    p.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    p.add_argument("--metric", default=os.getenv("MILVUS_METRIC_TYPE", "COSINE"))
    p.add_argument("--index-type", default=os.getenv("MILVUS_INDEX_TYPE", "HNSW"))
    p.add_argument("--multitenancy", default=os.getenv("ENABLE_MILVUS_MULTITENANCY_MODE", "false"))
    args, _ = p.parse_known_args()
    if args.chroma_path:
        os.environ["CHROMA_DATA_PATH"] = args.chroma_path
    if args.milvus_uri:
        os.environ["MILVUS_URI"] = args.milvus_uri
    if args.milvus_db is not None:
        os.environ["MILVUS_DB"] = args.milvus_db
    if args.milvus_token is not None:
        os.environ["MILVUS_TOKEN"] = args.milvus_token
    os.environ["MILVUS_METRIC_TYPE"] = args.metric
    os.environ["MILVUS_INDEX_TYPE"] = args.index_type
    os.environ["ENABLE_MILVUS_MULTITENANCY_MODE"] = str(args.multitenancy).lower()
    return args


_apply_cli_env()

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("migrate_chroma_to_milvus")

from open_webui.retrieval.vector.dbs.chroma import ChromaClient  # noqa: E402
from open_webui.retrieval.vector.dbs.milvus import MilvusClient  # noqa: E402

# open_webui is imported from the sibling Open WebUI checkout (it is not a
# pip-installable package). Resolve it via OPEN_WEBUI_BACKEND, falling back to
# the conventional relative location used in this workspace.
_BACKEND_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "open-webui-0.10.2", "backend")
)
BACKEND = os.environ.get("OPEN_WEBUI_BACKEND") or _BACKEND_DEFAULT
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def migrate(
    src_client: ChromaClient,
    dst_client: MilvusClient,
    batch_size: int = 500,
    collection_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Read every Chroma collection (with embeddings) and upsert into Milvus.

    Returns a summary dict: ``{"collections": [(name, n), ...], "total": int}``.
    """
    chroma = src_client.client
    collections = chroma.list_collections()
    names = [c.name if hasattr(c, "name") else c for c in collections]
    log.info(f"Found {len(names)} Chroma collection(s): {names}")

    summary: dict = {"collections": [], "total": 0}
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


def _all_ids(client, collection_name: str, dim: int = 384) -> set:
    """Return the set of all ids in a collection.

    Uses the high-level ``query`` (which reads flushed data, no top-k limit)
    rather than ``get``/``search``: the ORM ``Collection.query`` path misbehaves
    on Milvus Lite, and ``search`` requires a flush before freshly-upserted
    data becomes visible. We explicitly ``flush`` first, then paginate with
    ``limit``/``offset`` (Milvus caps those at 16384).
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


def _all_docs(client, collection_name: str, dim: int = 384) -> dict:
    """Return ``{id: text}`` for every entity, via ``query`` (no index needed).

    Unlike ``search``, ``query`` reads inserted data immediately after a flush,
    so it is safe to call right after a migration without waiting for the
    collection to be sealed/indexed.
    """
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


def verify(src_client: ChromaClient, dst_client: MilvusClient) -> bool:
    """Compare per-collection id sets between Chroma and Milvus."""
    chroma = src_client.client
    ok = True
    for c in chroma.list_collections():
        name = c.name if hasattr(c, "name") else c
        chroma_ids = set(chroma.get_collection(name).get(include=[])["ids"])
        sample = chroma.get_collection(name).get(limit=1, include=["embeddings"])
        emb = sample.get("embeddings")
        dim = len(emb[0]) if emb is not None and len(emb) > 0 else 384
        milvus_ids = _all_ids(dst_client, name, dim)
        match = chroma_ids == milvus_ids
        ok = ok and match
        log.info(
            f"[verify] {name}: chroma={len(chroma_ids)} milvus={len(milvus_ids)} "
            f"{'OK' if match else 'MISMATCH'}"
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate ChromaDB -> Milvus (offline)")
    parser.add_argument("--chroma-path", default=os.getenv("CHROMA_DATA_PATH"))
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI"))
    parser.add_argument("--milvus-db", default=os.getenv("MILVUS_DB"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--metric", default=os.getenv("MILVUS_METRIC_TYPE", "COSINE"))
    parser.add_argument("--index-type", default=os.getenv("MILVUS_INDEX_TYPE", "HNSW"))
    parser.add_argument(
        "--multitenancy",
        action="store_true",
        help="Target Milvus multitenancy mode (see MIGRATE_VECTOR.md caveats).",
    )
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
    if args.chroma_path:
        os.environ["CHROMA_DATA_PATH"] = args.chroma_path
    if args.milvus_uri:
        os.environ["MILVUS_URI"] = args.milvus_uri
    if args.milvus_db is not None:
        os.environ["MILVUS_DB"] = args.milvus_db
    if args.milvus_token is not None:
        os.environ["MILVUS_TOKEN"] = args.milvus_token
    os.environ["MILVUS_METRIC_TYPE"] = args.metric
    os.environ["MILVUS_INDEX_TYPE"] = args.index_type
    os.environ["ENABLE_MILVUS_MULTITENANCY_MODE"] = "true" if args.multitenancy else "false"

    src = ChromaClient()
    dst = MilvusClient()

    summary = migrate(
        src,
        dst,
        batch_size=args.batch,
        collection_filter=args.collections,
        dry_run=args.dry_run,
    )
    log.info(f"Summary: {summary['total']} vectors across {len(summary['collections'])} collection(s)")

    if args.verify and not args.dry_run:
        if not verify(src, dst):
            log.error("Verification FAILED: collection counts differ.")
            return 1
        log.info("Verification PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
