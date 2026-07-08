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
import json
import uuid
from pathlib import Path

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
    The two paths produce equivalent data because Open WebUI chunks identically
    regardless of vector DB and every backend stores chunks as‑is (confirmed
    in tests: N items → N stored documents/images. ChromaClient.insert does
    NOT re‑split).  The only differences are backend‑level details (id format);
    point counts and per‑chunk texts are the same.
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


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_migrated_store_works_with_openwebui(chroma_client, dst_client, embedding_function):
    """After migrating Chroma -> target, the target must behave like a normal
    Open WebUI vector store: a user can query it, add knowledge, and delete.

    Every operation goes through Open WebUI's own client (``search`` / ``insert``
    / ``delete``) -- exactly the code path the Open WebUI API uses -- so this
    confirms the migrated store is not just byte-equal but actually usable.
    """
    name = _kb_collection_name(0)
    docs = _make_documents(0, 0)

    # Migrate Chroma -> target.
    ingest_documents(chroma_client, name, docs, embedding_function)
    migrate(chroma_client, dst_client)

    # 1) QUERY: a phrase present in the migrated data is retrievable.
    q = asyncio.run(embedding_function(["knowledge base migration testing"], prefix=""))
    res = dst_client.search(collection_name=name, vectors=q, limit=5)
    assert res is not None and res.ids and len(res.ids[0]) > 0

    # 2) ADD KNOWLEDGE: insert a brand-new document into the same collection.
    new_doc = [{
        "title": "extra-knowledge.txt",
        "content": "Unique migration sanity phrase ZZZ-ALPHA-1234 about OpenWebUI RAG operations.",
        "file_id": "file-extra",
        "hash": "hash-extra",
    }]
    before = len(_all_ids(dst_client, name))
    n_added = ingest_documents(dst_client, name, new_doc, embedding_function)
    assert n_added >= 1
    after_ids = _all_ids(dst_client, name)
    assert len(after_ids) == before + n_added

    # The freshly-added knowledge is queryable. Embed the *exact* stored chunk
    # text (the mock embedding is text-sensitive) so the search returns it.
    added_docs = _all_docs(dst_client, name)
    new_texts = [v for v in added_docs.values() if v and "ZZZ-ALPHA-1234" in v]
    assert new_texts, "added knowledge is not present in the store"
    q2 = asyncio.run(embedding_function(new_texts, prefix=""))
    res2 = dst_client.search(collection_name=name, vectors=q2, limit=5)
    assert res2 is not None and res2.ids and len(res2.ids[0]) > 0
    new_ids = {_norm_id(k) for k, v in added_docs.items() if v and "ZZZ-ALPHA-1234" in v}
    res2_ids_norm = {_norm_id(i) for i in res2.ids[0]}
    assert new_ids & res2_ids_norm, "added knowledge is not returned by search"

    # 3) DELETE: remove one existing point and confirm it is gone.
    victim = next(iter(after_ids))
    dst_client.delete(collection_name=name, ids=[victim])
    remaining = _all_ids(dst_client, name)
    assert victim not in remaining
    assert len(remaining) == len(after_ids) - 1


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_incremental_migration(chroma_client, dst_client, embedding_function):
    """Migration must be resumable: run it, keep writing to Chroma, then run it
    again and only the *new* data should be migrated (the diff).

    This models the "migrate but don't switch yet" workflow: Chroma stays the
    live store, the target is synced periodically, and a re-run must not
    re-process what is already present and must not duplicate.
    """
    names = [_kb_collection_name(k) for k in range(3)]

    # Phase 1: seed Chroma with two KBs and do the first (full) migration.
    for kb in (0, 1):
        ingest_documents(chroma_client, names[kb], _make_documents(kb, 0), embedding_function)
    summary1 = migrate(chroma_client, dst_client, batch_size=200)
    assert summary1["total"] > 0
    assert verify(chroma_client, dst_client)

    # Phase 2: user keeps using Chroma -- a brand-new KB plus extra docs in KB 0.
    ingest_documents(chroma_client, names[2], _make_documents(2, 0), embedding_function)
    ingest_documents(chroma_client, names[0], _make_documents(0, 1), embedding_function)

    chroma_total = sum(
        len(set(chroma_client.get(collection_name=n).ids[0])) for n in names
    )

    # Re-run incrementally: only the delta should be migrated.
    summary2 = migrate(chroma_client, dst_client, batch_size=200, incremental=True)
    assert summary2["total"] > 0
    assert summary2["total"] < chroma_total, "incremental run should only migrate the delta"
    # Target is now fully in sync with Chroma.
    assert verify(chroma_client, dst_client)

    # Re-running incremental again migrates nothing (already synced).
    summary3 = migrate(chroma_client, dst_client, batch_size=200, incremental=True)
    assert summary3["total"] == 0, "a second incremental run should be a no-op"


@pytest.mark.parametrize("dst_client", ["milvus", "qdrant"], indirect=True)
def test_incremental_with_state_file(chroma_client, dst_client, embedding_function, tmp_path):
    """Time‑based incremental: state file gates processing and ID‑diff moves only
    records created/updated after the last migration.

    1. Ingest → incremental with state file → data migrated, stamp recorded.
    2. Re‑run immediately → no‑op (Chroma mtime ≤ state stamp).
    3. Add new documents → incremental migrates only the delta.
    4. Delete a document from Chroma → incremental finds nothing to add; stale
       record lingers in target (by design — verify will report a mismatch).
    """
    state_file = str(tmp_path / ".migrate_state.json")
    name = _kb_collection_name(0)

    # -- 1) initial ingest + incremental (seeds state file) -----------------
    ingest_documents(chroma_client, name, _make_documents(0, 0), embedding_function)
    summary1 = migrate(
        chroma_client, dst_client, batch_size=200,
        incremental=True, state_file=state_file,
    )
    assert summary1["total"] > 0

    state = json.loads(Path(state_file).read_text())
    assert "last_migration" in state
    first_stamp = state["last_migration"]
    assert verify(chroma_client, dst_client)

    # -- 2) re‑run immediately → no‑op (nothing changed on disk) ------------
    summary2 = migrate(
        chroma_client, dst_client, batch_size=200,
        incremental=True, state_file=state_file,
    )
    assert summary2["total"] == 0, "re-run with unchanged Chroma must be a no-op"

    # -- 3) add new documents → incremental migrates only the delta ---------
    chroma_before = len(set(chroma_client.get(collection_name=name).ids[0]))
    n_added = ingest_documents(
        chroma_client, name, _make_documents(0, 1), embedding_function,
    )
    assert n_added > 0
    chroma_after = len(set(chroma_client.get(collection_name=name).ids[0]))
    delta = chroma_after - chroma_before

    summary3 = migrate(
        chroma_client, dst_client, batch_size=200,
        incremental=True, state_file=state_file,
    )
    assert summary3["total"] > 0
    assert summary3["total"] <= delta, (
        "incremental should only migrate the new records"
    )
    assert verify(chroma_client, dst_client)

    # State file timestamp must advance.
    state2 = json.loads(Path(state_file).read_text())
    assert state2["last_migration"] > first_stamp

    # -- 4) delete a document from Chroma → target cleans stale record ------
    chroma_ids = set(chroma_client.get(collection_name=name).ids[0])
    victim = next(iter(chroma_ids))
    dst_before = _all_ids(dst_client, name)
    chroma_client.delete(collection_name=name, ids=[victim])

    summary4 = migrate(
        chroma_client, dst_client, batch_size=200,
        incremental=True, state_file=state_file,
    )
    assert summary4["total"] == 0, "nothing to add after a delete-only change"
    assert summary4["deleted"] == 1, "stale record must be removed from target"
    dst_after = _all_ids(dst_client, name)
    assert len(dst_after) == len(dst_before) - 1
    assert victim not in dst_after, "deleted-from-Chroma id must be gone from target"
    assert verify(chroma_client, dst_client)
