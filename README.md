# migrate-owui-rag

Offline migration of an **Open WebUI** ChromaDB vector store to a **Milvus
server** (standalone cluster: etcd + S3-compatible object store + Milvus).

The migration reuses Open WebUI's own `ChromaClient` / `MilvusClient` so
collection naming (`open_webui_` prefix + `-` → `_`), schema/dimension
creation, `process_metadata`, and the `MILVUS_TEXT_MAX_LENGTH` clamp all match
what the running application expects. No Open WebUI server is required.

## Layout

```
migrate-owui-rag/
├── migrate_chroma.py            # offline migration CLI (reuses open_webui clients)
├── docker-compose.milvus.yml    # etcd + RustFS + Milvus v2.6.14 for local testing
├── docker-compose.qdrant.yml    # Qdrant server for local testing
├── tests/
│   ├── mock_embedding_server.py  # Ollama-compatible /api/embed mock (deterministic)
│   ├── conftest.py               # fixtures: mock embed server, Chroma/Milvus/Qdrant clients
│   └── test_ingest_and_migrate.py
├── pyproject.toml
└── requirements.txt              # full, proven dependency set
```

## Dependencies on Open WebUI

This tool is not self-contained: it imports `open_webui.retrieval.vector.dbs.*`,
`open_webui.config`, etc. Those modules live in a checkout of Open WebUI and are
**not** pip-installable on their own. By default this project expects a sibling
checkout at `../open-webui-0.10.2/backend` (relative to this project's root).
Override the location with the `OPEN_WEBUI_BACKEND` environment variable:

```bash
export OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend
```

## Setup (uv)

```bash
cd migrate-owui-rag
uv venv --python 3.11
uv pip install -r requirements.txt      # proven set from the working env
# make open_webui importable in this venv (points at the sibling checkout):
echo "/abs/path/to/open-webui-0.10.2/backend" > .venv/lib/python3.11/site-packages/open_webui_backend.pth
```

Alternatively, set `OPEN_WEBUI_BACKEND` at runtime (the scripts also insert that
path into `sys.path` automatically).

## Stand up a Milvus server (local testing)

```bash
docker compose -f docker-compose.milvus.yml up -d
# -> Milvus gRPC on http://127.0.0.1:19530
```

Pin the image to the **same version** as `pymilvus` (here `v2.6.14`); mismatches
surface as `AllocTimestamp` "not implemented" gRPC errors. This stack uses
[RustFS](https://rustfs.com) as the S3-compatible object store in place of MinIO.

## Run the migration

The target backend is selected with `--dest` (`milvus`, the default, or
`qdrant`).

### To a Milvus server

```bash
export CHROMA_DATA_PATH=/path/to/open-webui/data/vector_db
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_DB=default
export MILVUS_METRIC_TYPE=COSINE          # MUST match Chroma's cosine
export ENABLE_MILVUS_MULTITENANCY_MODE=false
export OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend

# dry run: list collections + counts, write nothing
python migrate_chroma.py --dest milvus --dry-run

# real migration with verification
python migrate_chroma.py --dest milvus --batch 500 --verify
```

### To a Qdrant server

```bash
export CHROMA_DATA_PATH=/path/to/open-webui/data/vector_db
export QDRANT_URI=http://127.0.0.1:6333
export QDRANT_COLLECTION_PREFIX=open-webui
export OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend

# dry run
python migrate_chroma.py --dest qdrant --dry-run

# real migration with verification
python migrate_chroma.py --dest qdrant --batch 500 --verify
```

### CLI flags

| Flag | Env | Meaning |
| --- | --- | --- |
| `--dest` | `MIGRATE_DEST` | Target backend: `milvus` (default) or `qdrant` |
| `--chroma-path` | `CHROMA_DATA_PATH` | Chroma data directory |
| `--milvus-uri` | `MILVUS_URI` | Milvus server URI |
| `--milvus-db` | `MILVUS_DB` | Milvus database name |
| `--milvus-token` | `MILVUS_TOKEN` | Auth token (if enabled) |
| `--metric` | `MILVUS_METRIC_TYPE` | Distance metric (keep `COSINE`) |
| `--index-type` | `MILVUS_INDEX_TYPE` | e.g. `HNSW` |
| `--multitenancy` | `ENABLE_MILVUS_MULTITENANCY_MODE` | target multitenancy mode (Milvus) |
| `--qdrant-uri` | `QDRANT_URI` | Qdrant server URI |
| `--qdrant-api-key` | `QDRANT_API_KEY` | Qdrant API key (if enabled) |
| `--qdrant-prefix` | `QDRANT_COLLECTION_PREFIX` | Qdrant collection prefix (default `open-webui`) |
| `--qdrant-on-disk` | `QDRANT_ON_DISK` | store vectors on disk (Qdrant) |
| `--qdrant-prefer-grpc` | `QDRANT_PREFER_GRPC` | use gRPC transport (Qdrant) |
| `--batch` | — | upsert batch size (default 500) |
| `--collections` | — | migrate only these collection names |
| `--dry-run` | — | list collections/counts, write nothing |
| `--verify` | — | compare per-collection id sets afterwards |

The script targets the **standard** Milvus mode. Milvus *multitenancy* mode
routes every logical collection into shared physical collections, so a 1:1 copy
will not work there without replicating the routing logic.

## Switching Open WebUI to Milvus

1. Set `VECTOR_DB=milvus`, `MILVUS_URI=<target used during migration>`,
   `MILVUS_METRIC_TYPE=COSINE`, `ENABLE_MILVUS_MULTITENANCY_MODE=false`.
2. Start Open WebUI — it now uses the migrated Milvus store.
3. Keep the old `vector_db` (Chroma) until searches are verified, then archive.

Rollback is just `VECTOR_DB=chroma` again; the Chroma data is left untouched.

## Tests

The pytest suite emulates a user uploading **hundreds of documents across
multiple knowledge bases** (each KB == one vector collection) via the mock
embedding server, then migrates Chroma → the target store (Milvus **and**
Qdrant) offline and verifies fidelity. The two migration tests are
parametrized over both backends.

```bash
docker compose -f docker-compose.milvus.yml up -d
docker compose -f docker-compose.qdrant.yml up -d
WEBUI_SECRET_KEY=test OPEN_WEBUI_BACKEND=/abs/path/to/open-webui-0.10.2/backend \
    uv run pytest tests/test_ingest_and_migrate.py -v
```

The Milvus / Qdrant server fixtures auto-skip if Docker / docker compose is
unavailable, so the suite degrades gracefully (the embedding + ingest test
still runs).

### Notes / gotchas

- Milvus caps `limit`/`offset` at `16384`; verification enumerates with
  `flush()` + `query()` and paginates.
- Qdrant enumeration uses `scroll()` with an `offset` cursor (default page
  size `16384`).
- Freshly-upserted data is not `search`-visible until sealed/indexed, so
  verification enumerates via `flush()`+`query()` (Milvus) / `scroll()`
  (Qdrant), which read inserted data immediately, rather than `search()`.
- Chroma's wrapped `.get()` omits embeddings; the script calls the underlying
  `Collection.get(include=["embeddings", ...])` directly.
- Milvus server image version must match `pymilvus` (v2.6.14).

## Chunking analysis: migration vs. direct insert

A natural question is whether migrating an existing Chroma store into a target
backend yields the **same** vectors as if Open WebUI had been writing to that
target backend all along. In general it does **not**, and the reason is rooted
in how Open WebUI writes to Chroma, not in the migration script.

### What the migration preserves

The migration is a faithful, lossless copy of whatever Chroma already stored:
every Chroma point (id, document text, embedding, metadata) is re-created 1:1 in
the target via `upsert`. The existing `test_offline_migration_chroma_to_vector`
test asserts exactly this (id sets match, and verbatim document text matches).

### Where the two paths diverge

Open WebUI's `ChromaClient.insert` does **not** store the items you hand it
verbatim. It re-splits each item into several finer sub-documents. In the test
scenario (30 documents × 6 paragraphs, chunked with the langchain
`RecursiveCharacterTextSplitter` at `chunk_size=1000`, `chunk_overlap=100`):

| Path | Input chunks | Stored vectors | Chunk granularity |
| --- | --- | --- | --- |
| Direct insert into Qdrant/Milvus (`ingest_documents` → `client.insert`) | 60 | **60** | one vector per langchain chunk |
| Insert into Chroma, then migrate to Qdrant/Milvus | 60 → Chroma → migrate | **~180** | Chroma re-split each item into ~3 finer sub-documents |

So a Chroma → Qdrant migration produces roughly **3× more, shorter chunks**
than a direct Qdrant insert of the same source documents, and the individual
chunk texts differ (Chroma's fragments are finer than the langchain chunks).
This was confirmed empirically: 60 langchain chunks produced **180** documents
in Chroma (lengths ~400–810 chars), whereas the direct Qdrant insert kept the
60 original langchain chunks.

### Why this is expected, not a bug

- The migration copies Chroma's *actual* stored layout. Reconstructing the
  finer Chroma chunks in the target is the **correct** behaviour — it means the
  migrated store behaves exactly like the Chroma store it came from (same
  search results, same retrieval).
- The difference originates entirely in the **source** backend (Chroma's
  client re-chunks on write). A direct insert into Qdrant/Milvus bypasses
  Chroma, so it keeps the coarser langchain chunking.
- The exact ratio depends on document length and the configured chunk size /
  overlap; it is not a fixed constant.

### How the test accounts for this

Because ids, counts and per-chunk vectors legitimately differ between the two
paths, `test_direct_insert_vs_chroma_migration` does **not** assert
byte-identical collections. Instead it groups each store's points by
`metadata.hash` (the original source document) and verifies that **every
original line of text is recoverable from both stores**. Both paths preserve
the same underlying knowledge — they just slice it differently.

**Takeaway:** prefer migrating Chroma → target over standing up a fresh target
store and re-ingesting, *if* you want the new store to match the existing
Chroma-backed experience. If instead you re-ingest the same files directly into
the target backend, you will get coarser chunks and therefore different search
behaviour than the Chroma system you are replacing.
