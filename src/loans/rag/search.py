"""The search for the offer agent's RAG: the chunks closest in meaning to a
query, from the index in Postgres (loans.rag.index).

The query is embedded with the same model as the chunks (with the
"search_query: " prefix), and pgvector returns the nearest chunks by cosine
distance (the <=> operator, served by the HNSW index). Only chunks embedded
with the current model are compared: embeddings from different models are
different "number languages". If the index has none (empty, or built with
another model), the search stops with IndexNotReady, a system error the agent
never sees, rather than returning nothing.

There is no cut-off on similarity yet: the closest chunks are always
returned and the agent judges whether they answer the customer's question.
A cut-off can be calibrated later by the retrieval evals.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from loans.rag.embeddings import EMBEDDING_MODEL, embed_query

# How many chunks a search returns: about 800 words, enough to include the
# right section when it ranks second or third, small enough for the context.
TOP_K = 4


class IndexNotReady(RuntimeError):
    """The index has no chunks embedded with the current model: rebuild it."""


class SearchResult(BaseModel):
    chunk_id: str
    document: str
    title: str
    text: str
    source: str
    # Cosine distance to the query: 0 is the same meaning, larger is further.
    distance: float


def search(conn: psycopg.Connection, query: str, k: int = TOP_K, embeddings=None) -> list[SearchResult]:
    """The k chunks closest in meaning to the query, closest first."""
    register_vector(conn)
    query_vector = Vector(embed_query(query, embeddings))
    rows = conn.execute(
        "SELECT chunk_id, document, title, text, source, embedding <=> %(q)s AS distance"
        " FROM rag_chunks WHERE embedding_model = %(model)s"
        " ORDER BY embedding <=> %(q)s LIMIT %(k)s",
        {"q": query_vector, "model": EMBEDDING_MODEL, "k": k},
    ).fetchall()
    if not rows:
        raise IndexNotReady(
            f"the search index has no chunks embedded with {EMBEDDING_MODEL}; "
            "rebuild it with: uv run python -m loans.rag.index"
        )
    fields = ["chunk_id", "document", "title", "text", "source", "distance"]
    return [SearchResult(**dict(zip(fields, row, strict=True))) for row in rows]
