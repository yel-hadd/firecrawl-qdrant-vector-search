"""Offline test: local chunker + local embeddings + in-memory Qdrant.

No API keys needed (OPENAI_API_KEY is unset so FastEmbed is used).
"""

from src.chunk import chunk_local
from src.crawl import Page
from src.index import delete_stale, upsert_chunks
from src.search import hybrid_search, semantic_search

PAGES = [
    Page(url="https://example.com/auth", markdown="# Authentication\n\n## API keys\n\nCreate an API key in the dashboard and send it in the `api-key` header.\n\n## Login errors\n\nA 401 response means the key is missing or revoked."),
    Page(url="https://example.com/collections", markdown="# Collections\n\nA collection stores points with vectors and a JSON payload.\n\n## Payload indexing\n\nIndex payload fields you filter on."),
]


def test_index_and_search(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from qdrant_client import QdrantClient, models

    client = QdrantClient(location=":memory:")
    chunks = chunk_local(PAGES)
    assert upsert_chunks(client, chunks, name="t") == len(chunks)

    assert semantic_search(client, "why do I get a 401 error", name="t")[0]["url"] == "https://example.com/auth"
    assert hybrid_search(client, "api-key header", name="t")[0]["url"] == "https://example.com/auth"

    # Re-indexing the same chunks overwrites instead of duplicating.
    upsert_chunks(client, chunks, name="t")
    assert client.count("t").count == len(chunks)

    # An edited page drops its old points.
    edited = [Page(url=PAGES[0].url, markdown=PAGES[0].markdown + "\n\n## Rotation\n\nRotate keys every 90 days.")]
    new_chunks = chunk_local(edited)
    upsert_chunks(client, new_chunks, name="t")
    delete_stale(client, new_chunks, name="t")
    url_filter = models.Filter(must=[models.FieldCondition(key="url", match=models.MatchValue(value=PAGES[0].url))])
    auth_points = client.count("t", count_filter=url_filter).count
    assert auth_points == len(new_chunks)
