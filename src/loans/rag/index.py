"""The search index for the offer agent's RAG: the chunks and their
embeddings, stored in Postgres with pgvector.

The chunks live in the rag_chunks table of the bank's database, next to the
cases (see schema.sql): each with its title, text, source, embedding, and
the name of the model that made the embedding, so a question is never
compared with embeddings from another model.

Building the index is a full rebuild: every chunk is embedded first (about
10 seconds for the hundred or so chunks), then the table's contents are
replaced in one transaction. A failure while embedding leaves the old index
untouched, and a search never sees a half-built index. With this many
chunks, a rebuild is simpler than updating only what changed.

Rebuild after a change to the documents (Postgres and Ollama running):

    uv run python -m loans.rag.index
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from loans.db.postgres_store import connect, create_tables, database_conninfo
from loans.rag.chunks import Chunk, all_chunks
from loans.rag.embeddings import EMBEDDING_MODEL, embed_documents


def build_index(conn: psycopg.Connection, chunks: list[Chunk] | None = None, embeddings=None) -> int:
    """Embed the chunks (by default every chunk of the documents) and make
    them the whole index. Returns how many chunks were stored."""
    chunks = all_chunks() if chunks is None else chunks
    vectors = embed_documents([c.text_to_embed for c in chunks], embeddings)
    register_vector(conn)
    rows = [
        (c.chunk_id, c.document, c.title, c.text, c.source, Vector(vector), EMBEDDING_MODEL)
        for c, vector in zip(chunks, vectors, strict=True)
    ]
    with conn.transaction():
        conn.execute("DELETE FROM rag_chunks")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rag_chunks (chunk_id, document, title, text, source, embedding,"
                " embedding_model) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
    return len(rows)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    with connect(database_conninfo()) as conn:
        create_tables(conn)  # only what is missing
        count = build_index(conn)
    print(f"indexed {count} chunks with {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
