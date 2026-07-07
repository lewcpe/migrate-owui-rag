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
from migrate_chroma import migrate, verify, _all_ids, _all_docs

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
