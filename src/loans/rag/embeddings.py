"""Embeddings for the offer agent's RAG: turning text into numbers that
capture its meaning, so the search can find the chunks closest to a
question even when the words differ ("time off" and "vacation").

The model is nomic-embed-text v1.5, run locally through Ollama like the
agent's LLM, so no text leaves the laptop. It turns a text into 768 numbers.
It expects a short prefix saying what the text is for: "search_document: "
for the chunks that are searched, "search_query: " for the question; using
them makes the search noticeably better. Only the embedded text gets the
prefix; the chunk itself is stored and shown without it.

Install the model once with:

    ollama pull nomic-embed-text:v1.5
"""

from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text:v1.5"
DIMENSIONS = 768

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def make_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def embed_documents(texts: list[str], embeddings: OllamaEmbeddings | None = None) -> list[list[float]]:
    """The embeddings of the texts to be searched, in order."""
    embeddings = embeddings or make_embeddings()
    return embeddings.embed_documents([DOCUMENT_PREFIX + text for text in texts])


def embed_query(question: str, embeddings: OllamaEmbeddings | None = None) -> list[float]:
    """The embedding of a question, to compare with the documents'."""
    embeddings = embeddings or make_embeddings()
    return embeddings.embed_query(QUERY_PREFIX + question)
