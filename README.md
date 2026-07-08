# migrate-owui-rag

Offline migration of an **Open WebUI** ChromaDB vector store to another vector
backend. Currently supports **Milvus** and **Qdrant**.

Reuses Open WebUI's own clients so collection naming, schemas, and metadata
processing match the running application. No Open WebUI server is required.

## Setup

This tool imports from a checkout of Open WebUI. Point `OPEN_WEBUI_BACKEND` at
the `backend/` directory (default: `../open-webui-0.10.2/backend`):

```bash
export OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend

cd migrate-owui-rag
uv venv --python 3.11
uv pip install -r requirements.txt
echo "/abs/path/to/open-webui-0.10.2/backend" > .venv/lib/python3.11/site-packages/open_webui_backend.pth
```

## Local test servers

```bash
docker compose -f docker-compose.milvus.yml up -d  # gRPC on :19530
docker compose -f docker-compose.qdrant.yml up -d  # REST on :6333
```

The Milvus image must match `pymilvus` (v2.6.14). The stack uses RustFS as the
S3-compatible object store.

## Usage

Target backend is selected with `--dest` (`milvus` default, or `qdrant`).

### Milvus

```bash
export CHROMA_DATA_PATH=/path/to/open-webui/data/vector_db
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_METRIC_TYPE=COSINE

python migrate_chroma.py --dest milvus --dry-run
python migrate_chroma.py --dest milvus --batch 500 --verify
```

### Qdrant

```bash
export CHROMA_DATA_PATH=/path/to/open-webui/data/vector_db
export QDRANT_URI=http://127.0.0.1:6333

python migrate_chroma.py --dest qdrant --dry-run
python migrate_chroma.py --dest qdrant --batch 500 --verify
```

### CLI flags

| Flag | Env | Meaning |
| --- | --- | --- |
| `--dest` | `MIGRATE_DEST` | `milvus` (default) or `qdrant` |
| `--chroma-path` | `CHROMA_DATA_PATH` | Chroma data directory |
| `--batch` | — | upsert batch size (default 500) |
| `--collections` | — | migrate only these collection names |
| `--dry-run` | — | list collections/counts, write nothing |
| `--incremental` | — | time-aware sync: adds missing + deletes stale records |
| `--state-file` | — | state file for incremental (default `.migrate_state.json`) |
| `--verify` | — | compare per-collection id sets afterwards |

**Backend-specific flags** (Milvus): `--milvus-uri`, `--milvus-db`,
`--milvus-token`, `--metric`, `--index-type`, `--multitenancy`.

**Backend-specific flags** (Qdrant): `--qdrant-uri`, `--qdrant-api-key`,
`--qdrant-prefix`, `--qdrant-on-disk`, `--qdrant-prefer-grpc`.

All flags also read from the corresponding environment variables shown above.

### Incremental mode

`--incremental` writes a state file recording the last migration timestamp.
Subsequent runs:

- **Skip everything** if Chroma data hasn't changed on disk since the last run.
- Otherwise do a per-collection ID-diff: add missing records and delete stale
  ones from the target. This covers creates, re-uploads (new UUIDs), and
  deletes.

```bash
python migrate_chroma.py --dest milvus --incremental --verify
```

Delete `.migrate_state.json` to force a full migration next time.

## Switching Open WebUI

1. Set `VECTOR_DB=milvus` (or `qdrant`) with the URI used during migration.
2. Restart Open WebUI — it uses the migrated store.
3. Archive the old Chroma `vector_db` once verified.

Roll back by setting `VECTOR_DB=chroma`; the Chroma data is left untouched.

## Tests

```bash
docker compose -f docker-compose.milvus.yml up -d
docker compose -f docker-compose.qdrant.yml up -d

WEBUI_SECRET_KEY=test OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend \
    uv run pytest tests/test_ingest_and_migrate.py -v
```

Server fixtures auto-skip when Docker is unavailable.

### Gotchas

- Milvus caps `limit`/`offset` at 16384; verification paginates with
  `flush()` + `query()`.
- Qdrant enumeration uses `scroll()` with an offset cursor.
- Freshly-upserted data is not `search`-visible until indexed — verification
  uses `query()` / `scroll()` instead.
- Chroma's wrapped `.get()` omits embeddings; the script calls the underlying
  `Collection.get(include=["embeddings", ...])` directly.
- Milvus server image must match `pymilvus` (v2.6.14).
