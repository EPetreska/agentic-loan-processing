"""Tests for the embeddings. The prefix tests need nothing; the model tests
need Ollama running with nomic-embed-text and are skipped without it."""

import math

import pytest

from loans.rag.chunks import all_chunks
from loans.rag.embeddings import DIMENSIONS, embed_documents, embed_query, make_embeddings


class Recorder:
    """Stands in for the model and records what it was given."""

    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen += texts
        return [[0.0] * DIMENSIONS for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [0.0] * DIMENSIONS


def test_documents_and_questions_get_their_prefixes():
    recorder = Recorder()
    embed_documents(["Late payments"], recorder)
    embed_query("How much is the late fee?", recorder)
    assert recorder.seen == ["search_document: Late payments", "search_query: How much is the late fee?"]


@pytest.fixture(scope="module")
def model():
    embeddings = make_embeddings()
    try:
        embeddings.embed_query("ping")
    except Exception as e:  # Ollama not running, or the model not installed
        pytest.skip(f"Ollama with nomic-embed-text not reachable: {e}")
    return embeddings


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def test_the_model_gives_768_numbers(model):
    assert len(embed_query("What is APR?", model)) == DIMENSIONS


def test_a_late_fee_question_finds_the_banks_late_payments_first(model):
    # The bank's terms and the CFPB's answer about late fees on car loans,
    # a close cousin of the question that must not win.
    chunks = [c for c in all_chunks() if c.document == "bank" or c.chunk_id.startswith("cfpb-839-")]
    vectors = embed_documents([c.text_to_embed for c in chunks], model)
    question = embed_query("How much is the late fee?", model)
    best = max(zip(vectors, chunks), key=lambda pair: cosine(question, pair[0]))[1]
    assert best.document == "bank" and best.title == "Late payments"
