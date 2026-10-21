"""Tests for the RAG index in Postgres with pgvector. They need Postgres
running (docker compose up -d); the last one also needs Ollama."""

import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector

from loans.rag.chunks import all_chunks
from loans.rag.embeddings import DIMENSIONS, EMBEDDING_MODEL, embed_query, make_embeddings
from loans.rag.index import build_index


class FakeModel:
    """Stands in for the embedding model: a different, non-zero vector for
    each text, so the tests are fast and need no Ollama."""

    def embed_documents(self, texts):
        return [[1.0 if i == n % DIMENSIONS else 0.0 for i in range(DIMENSIONS)] for n in range(len(texts))]


class BrokenModel:
    def embed_documents(self, texts):
        raise RuntimeError("the model stopped")


def rows(conn):
    return conn.execute(
        "SELECT chunk_id, document, title, source, embedding_model FROM rag_chunks ORDER BY chunk_id"
    ).fetchall()


def test_every_chunk_is_stored_with_its_source(pg_conn):
    chunks = all_chunks()
    assert build_index(pg_conn, chunks, FakeModel()) == len(chunks)
    stored = {row[0]: row for row in rows(pg_conn)}
    assert set(stored) == {c.chunk_id for c in chunks}
    late = next(c for c in chunks if c.title == "Late payments")
    assert stored[late.chunk_id][1:4] == ("bank", "Late payments", late.source)


def test_the_model_is_recorded_with_each_chunk(pg_conn):
    build_index(pg_conn, all_chunks()[:5], FakeModel())
    assert {row[4] for row in rows(pg_conn)} == {EMBEDDING_MODEL}


def test_a_rebuild_replaces_the_old_index(pg_conn):
    chunks = all_chunks()
    build_index(pg_conn, chunks, FakeModel())
    build_index(pg_conn, chunks[:3], FakeModel())  # as after a document shrank
    assert [row[0] for row in rows(pg_conn)] == sorted(c.chunk_id for c in chunks[:3])


def test_a_failed_rebuild_leaves_the_old_index(pg_conn):
    chunks = all_chunks()[:5]
    build_index(pg_conn, chunks, FakeModel())
    with pytest.raises(RuntimeError):
        build_index(pg_conn, all_chunks(), BrokenModel())
    assert len(rows(pg_conn)) == 5


def test_the_embeddings_are_stored_as_vectors(pg_conn):
    build_index(pg_conn, all_chunks()[:2], FakeModel())
    register_vector(pg_conn)
    (vector,) = pg_conn.execute("SELECT embedding FROM rag_chunks LIMIT 1").fetchone()
    assert isinstance(vector, Vector) and len(vector.to_list()) == DIMENSIONS


def test_postgres_finds_the_banks_late_payments_for_a_late_fee_question(pg_conn):
    # With the real model: the nearest chunk by cosine distance (pgvector's
    # <=> operator) to the question is the bank's own late payments section.
    model = make_embeddings()
    try:
        model.embed_query("ping")
    except Exception as e:
        pytest.skip(f"Ollama with nomic-embed-text not reachable: {e}")
    build_index(pg_conn, all_chunks(), model)
    register_vector(pg_conn)
    question = Vector(embed_query("How much is the late fee?", model))
    (title, document) = pg_conn.execute(
        "SELECT title, document FROM rag_chunks ORDER BY embedding <=> %s LIMIT 1", (question,)
    ).fetchone()
    assert (title, document) == ("Late payments", "bank")
