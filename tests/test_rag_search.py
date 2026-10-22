"""Tests for the RAG search in Postgres. They need Postgres running; the last
ones also need Ollama and are skipped without it."""

import zlib

import pytest

from loans.rag.chunks import all_chunks
from loans.rag.embeddings import DIMENSIONS, DOCUMENT_PREFIX, QUERY_PREFIX, make_embeddings
from loans.rag.index import build_index
from loans.rag.search import IndexNotReady, search


def _vector(text: str) -> list[float]:
    """A fixed, non-zero vector per text: the same text, the same vector."""
    vector = [0.0] * DIMENSIONS
    vector[zlib.crc32(text.encode()) % DIMENSIONS] = 1.0
    vector[(zlib.crc32(text.encode()) // DIMENSIONS) % DIMENSIONS] += 0.5
    return vector


class FakeModel:
    """Stands in for the embedding model: a question equal to a chunk's text
    gets that chunk's vector, so it is found first."""

    def embed_documents(self, texts):
        return [_vector(t.removeprefix(DOCUMENT_PREFIX)) for t in texts]

    def embed_query(self, text):
        return _vector(text.removeprefix(QUERY_PREFIX))


@pytest.fixture
def indexed(pg_conn):
    chunks = all_chunks()
    build_index(pg_conn, chunks, FakeModel())
    return pg_conn, chunks


def test_the_closest_chunks_come_first_with_their_sources(indexed):
    conn, chunks = indexed
    late = next(c for c in chunks if c.title == "Late payments")
    results = search(conn, late.text_to_embed, embeddings=FakeModel())
    assert len(results) == 4
    assert results[0].chunk_id == late.chunk_id and results[0].distance == pytest.approx(0)
    assert (results[0].title, results[0].source, results[0].text) == (late.title, late.source, late.text)
    assert [r.distance for r in results] == sorted(r.distance for r in results)


def test_k_sets_how_many_come_back(indexed):
    conn, _ = indexed
    assert len(search(conn, "anything", k=2, embeddings=FakeModel())) == 2


def test_chunks_from_another_model_are_never_used(indexed):
    conn, _ = indexed
    conn.execute("UPDATE rag_chunks SET embedding_model = 'another-model'")
    with pytest.raises(IndexNotReady, match="rebuild it"):
        search(conn, "late fee", embeddings=FakeModel())


def test_an_empty_index_is_a_clear_error(pg_conn):
    with pytest.raises(IndexNotReady, match="uv run python -m loans.rag.index"):
        search(pg_conn, "late fee", embeddings=FakeModel())


@pytest.fixture(scope="module")
def model():
    embeddings = make_embeddings()
    try:
        embeddings.embed_query("ping")
    except Exception as e:
        pytest.skip(f"Ollama with nomic-embed-text not reachable: {e}")
    return embeddings


@pytest.fixture
def real_index(pg_conn, model):
    build_index(pg_conn, all_chunks(), model)
    return pg_conn


def test_with_the_real_model_the_banks_terms_come_first(real_index, model):
    # One index build (about 11 seconds) for both questions.
    [best, *_] = search(real_index, "How much is the late fee?", embeddings=model)
    assert (best.document, best.title) == ("bank", "Late payments")

    # The bank's "Paying early" ranks before the CFPB's mortgage page.
    results = search(real_index, "Is there a penalty if I pay my loan off early?", embeddings=model)
    bank = next(i for i, r in enumerate(results) if r.title == "Paying early")
    assert all(i > bank for i, r in enumerate(results) if r.chunk_id.startswith("cfpb-1957-"))
