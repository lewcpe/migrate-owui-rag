"""End-to-end offline tests for Chroma ingestion + Milvus migration.

These emulate a user uploading many documents into several Open WebUI
knowledge bases (each knowledge base == one vector collection) using a mock
Ollama embedding server, then migrating the resulting Chroma store to Milvus
fully offline.

Run from the backend directory:

    pytest tests/test_ingest_and_migrate.py -v

or, with the uv environment:

    uv run --python .venv pytest tests/test_ingest_and_migrate.py -v
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from tests.conftest import ingest_documents
from migrate_chroma import migrate, verify, _all_ids, _all_docs, _backend_of

# Test scenario: 5 knowledge bases, ~30 documents each -> hundreds of docs, and
# because each document is multi-paragraph it yields several chunks, so the
# vector stores end up with well over a thousand items total.
NUM_KB = 5
DOCS_PER_KB = 30
PARAGRAPHS_PER_DOC = 6


def _make_documents(kb_index: int, doc_index: int) -> list[dict]:
    docs = []
    for d in range(DOCS_PER_KB):
        paragraphs = [
            (
                f"Knowledge base {kb_index} document {d}, paragraph {p}. "
                f"OpenWebUI migration testing requires realistic multi-chunk text "
                f"so that the vector store accumulation is representative. "
                f"uuid={uuid.uuid4().hex}"
            )
            for p in range(PARAGRAPHS_PER_DOC)
        ]
        content = "\n\n".join(paragraphs)
        docs.append(
            {
                "title": f"kb{kb_index}_doc{d}.txt",
                "content": content,
                "file_id": f"file-{kb_index}-{d}",
                "hash": f"hash-{kb_index}-{d}",
            }
        )
    return docs


def _kb_collection_name(kb_index: int) -> str:
    # Use a hyphenated name to also exercise Milvus' '-' -> '_' remapping.
    return f"kb-{kb_index}"


def test_mock_server_dimension(mock_embedding_server):
    from tests.mock_embedding_server import embed_text

    v = embed_text("hello world", dim=384)
    assert len(v) == 384
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    # deterministic
    assert embed_text("hello world", dim=384) == v


def test_ingest_multiple_knowledge_bases(chroma_client, embedding_function):
    total_items = 0
    for kb in range(NUM_KB):
        name = _kb_collection_name(kb)
        docs = _make_documents(kb, 0)
        n = ingest_documents(chroma_client, name, docs, embedding_function)
        assert n > 0
        total_items += n

    # Hundreds of documents across multiple KBs -> many hundreds of vectors.
    assert total_items >= NUM_KB * DOCS_PER_KB
    print(f"\nIngested {total_items} vectors across {NUM_KB} knowledge bases")

    # Every KB collection exists in Chroma.
    for kb in range(NUM_KB):
        assert chroma_client.has_collection(_kb_collection_name(kb))

    # Sanity: a semantic-ish search returns results.
    q = asyncio.run(embedding_function(["knowledge base migration testing"], prefix=""))
    res = chroma_client.search(collection_name=_kb_collection_name(0), vectors=q, limit=3)
    assert res is not None and res.ids and len(res.ids[0]) > 0


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_offline_migration_chroma_to_vector(
    chroma_client, dst_client, embedding_function
):
    # 1) Ingest into Chroma (multiple KBs, hundreds of docs).
    for kb in range(NUM_KB):
        ingest_documents(
            chroma_client, _kb_collection_name(kb), _make_documents(kb, 0), embedding_function
        )

    # 2) Migrate offline Chroma -> target vector store.
    summary = migrate(chroma_client, dst_client, batch_size=200)
    assert summary["total"] > 0

    # 3) Verify every collection landed in the target with matching counts.
    assert verify(chroma_client, dst_client)

    # 4) Spot-check content fidelity on one collection via enumeration
    #    (not ``search``): freshly-upserted data is not searchable until it is
    #    sealed/indexed, whereas flush+query / scroll sees it immediately.
    name = _kb_collection_name(0)
    chroma_res = chroma_client.get(collection_name=name)
    dst_ids = _all_ids(dst_client, name)
    assert set(chroma_res.ids[0]) == dst_ids

    mtexts = _all_docs(dst_client, name)
    ctexts = {i: t for i, t in zip(chroma_res.ids[0], chroma_res.documents[0])}
    assert ctexts == mtexts

    # 5) Target search works post-migration.
    q = asyncio.run(embedding_function(["knowledge base migration testing"], prefix=""))
    res = dst_client.search(collection_name=name, vectors=q, limit=3)
    assert res is not None and res.ids and len(res.ids[0]) > 0


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_migration_is_idempotent(chroma_client, dst_client, embedding_function):
    ingest_documents(
        chroma_client, _kb_collection_name(0), _make_documents(0, 0), embedding_function
    )
    migrate(chroma_client, dst_client)
    first = dst_client.get(collection_name=_kb_collection_name(0))
    first_n = len(first.ids[0]) if first.ids else 0

    # Re-running the migration must not duplicate (upsert by id).
    migrate(chroma_client, dst_client)
    second = dst_client.get(collection_name=_kb_collection_name(0))
    second_n = len(second.ids[0]) if second.ids else 0
    assert first_n == second_n


def _norm_id(pid) -> str:
    """Normalize a point id to a compact hex string (Qdrant returns UUID ids)."""
    if isinstance(pid, uuid.UUID):
        return pid.hex
    if isinstance(pid, str):
        try:
            return uuid.UUID(pid).hex
        except ValueError:
            return pid
    return str(pid)


def _dst_content(client, collection_name: str) -> dict:
    """Return ``{id: {"text", "vector": [rounded], "metadata"}}`` for a store.

    Reads vectors too, so the direct-insert vs migrate paths can be compared
    content-wise (text + vector + metadata) rather than just by id/count.
    """
    if _backend_of(client) == "qdrant":
        full = f"{client.collection_prefix}_{collection_name}"
        out: dict = {}
        offset = None
        while True:
            pts, offset = client.client.scroll(
                collection_name=full, limit=16384, offset=offset, with_vectors=True
            )
            for p in pts:
                out[_norm_id(p.id)] = {
                    "text": (p.payload or {}).get("text"),
                    "vector": [round(float(v), 6) for v in (p.vector or [])],
                    "metadata": (p.payload or {}).get("metadata"),
                }
            if offset is None:
                break
        return out

    # Milvus: flush so freshly-upserted data is visible, then query with vectors.
    full = f"{client.collection_prefix}_{collection_name.replace('-', '_')}"
    try:
        client.client.flush(collection_name=full)
    except Exception:
        pass
    out = {}
    offset = 0
    while True:
        res = client.client.query(
            collection_name=full,
            filter="",
            output_fields=["id", "data", "vector", "metadata"],
            limit=16384,
            offset=offset,
        )
        if not res:
            break
        for r in res:
            data = r.get("data") or {}
            text = data.get("text") if isinstance(data, dict) else None
            out[_norm_id(r.get("id"))] = {
                "text": text,
                "vector": [round(float(v), 6) for v in (r.get("vector") or [])],
                "metadata": r.get("metadata"),
            }
        if len(res) < 16384:
            break
        offset += 16384
    return out


def _lines_by_hash(content: dict) -> dict:
    """For each original document (keyed by ``metadata.hash``), collect the set of
    stripped, non-empty lines present across all of its stored chunks.

    This lets the two ingestion scenarios be compared at the *document* level
    instead of the *chunk* level: different backends chunk text differently, so
    point ids, counts and per-chunk vectors legitimately differ, but the
    underlying knowledge (every original line) must be preserved in both.
    """
    out: dict = {}
    for c in content.values():
        meta = c.get("metadata") or {}
        h = meta.get("hash")
        if h is None:
            continue
        lines = {ln.strip() for ln in (c.get("text") or "").splitlines() if ln.strip()}
        out.setdefault(h, set()).update(lines)
    return out


def _orig_lines(docs: list[dict]) -> dict:
    out = {}
    for d in docs:
        out[d["hash"]] = {ln.strip() for ln in d["content"].splitlines() if ln.strip()}
    return out


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_direct_insert_vs_chroma_migration(chroma_client, dst_client, embedding_function):
    """Inserting the same docs directly into the target store vs migrating them
    from Chroma must yield the *same knowledge*.

    Scenario A: ingest the same documents straight into the target store via
    Open WebUI's client (``insert``).
    Scenario B: ingest the same documents into Chroma, then migrate to the
    target store via ``migrate``.

    The two paths do NOT produce byte-identical collections: Chroma's client
    re-chunks each item into finer pieces, so point ids, counts and per-chunk
    vectors differ between the two stores. What must match is the underlying
    content -- every original document's text must be fully recoverable from
    both stores (checked line-by-line, grouped by ``metadata.hash``).
    """
    name_direct = "kb-direct"
    name_migrated = "kb-migrated"
    docs = _make_documents(0, 0)

    # Scenario A: directly into the target store.
    n_direct = ingest_documents(dst_client, name_direct, docs, embedding_function)
    assert n_direct > 0

    # Scenario B: Chroma first, then migrate.
    ingest_documents(chroma_client, name_migrated, docs, embedding_function)
    migrate(chroma_client, dst_client)

    direct = _lines_by_hash(_dst_content(dst_client, name_direct))
    migrated = _lines_by_hash(_dst_content(dst_client, name_migrated))
    original = _orig_lines(docs)

    assert set(direct) == set(migrated) == set(original), "source document sets differ"
    for h in original:
        assert original[h] <= direct[h], f"direct insert lost lines for doc {h}"
        assert original[h] <= migrated[h], f"chroma->target migration lost lines for doc {h}"
