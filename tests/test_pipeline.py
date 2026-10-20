"""Offline tests: in-memory Qdrant, local embeddings, a mocked Apify client.

No API keys needed (OPENAI_API_KEY is unset so FastEmbed is used).
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from src.chunk import chunk_local, chunk_with_apify, document_hash
from src.crawl import Page
from src.index import delete_stale, indexed_hashes, upsert_chunks
from src.search import hybrid_search, semantic_search

# Recorded output of a real Markdown RAG Chunker run on PAGES[0].
RECORDED = json.loads((Path(__file__).parent / "fixtures" / "chunker_output.json").read_text())

PAGES = [
    Page(url="https://example.com/auth", markdown="# Authentication\n\n## API keys\n\nCreate an API key in the dashboard and send it in the `api-key` header.\n\n## Login errors\n\nA 401 response means the key is missing or revoked."),
    Page(url="https://example.com/collections", markdown="# Collections\n\nA collection stores points with vectors and a JSON payload.\n\n## Payload indexing\n\nIndex payload fields you filter on."),
]


@pytest.fixture(autouse=True)
def local_embeddings(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def url_count(client, url):
    url_filter = models.Filter(must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))])
    return client.count("t", count_filter=url_filter).count


def test_index_and_search():
    client = QdrantClient(location=":memory:")
    chunks = chunk_local(PAGES)
    assert upsert_chunks(client, chunks, name="t") == len(chunks)

    assert semantic_search(client, "why do I get a 401 error", name="t")[0]["url"] == "https://example.com/auth"
    assert hybrid_search(client, "api-key header", name="t")[0]["url"] == "https://example.com/auth"

    # Re-indexing the same chunks overwrites instead of duplicating.
    upsert_chunks(client, chunks, name="t")
    assert client.count("t").count == len(chunks)

    # An edited page drops its old points; the other page is untouched.
    edited = [Page(url=PAGES[0].url, markdown=PAGES[0].markdown + "\n\n## Rotation\n\nRotate keys every 90 days.")]
    new_chunks = chunk_local(edited)
    upsert_chunks(client, new_chunks, name="t")
    delete_stale(client, new_chunks, name="t")
    assert url_count(client, PAGES[0].url) == len(new_chunks)
    assert url_count(client, PAGES[1].url) > 0


def test_local_hash_matches_actor_hash():
    assert document_hash(PAGES[0].markdown) == RECORDED[0]["document_hash"]


def test_unchanged_pages_are_detected():
    client = QdrantClient(location=":memory:")
    assert indexed_hashes(client, [p.url for p in PAGES], name="t") == {}
    upsert_chunks(client, chunk_local(PAGES), name="t")

    known = indexed_hashes(client, [p.url for p in PAGES], name="t")
    edited = Page(url=PAGES[1].url, markdown=PAGES[1].markdown + "\n\nNew paragraph.")
    todo = [p for p in [PAGES[0], edited] if known.get(p.url) != {document_hash(p.markdown)}]
    assert todo == [edited]


class FakeApify:
    """Mimics ApifyClientAsync: actor().call() and dataset().iterate_items()."""

    def __init__(self, items_by_url, status="SUCCEEDED"):
        self.items_by_url, self.status = items_by_url, status
        self.calls, self.running, self.peak = [], 0, 0

    def actor(self, actor_id):
        async def call(run_input, max_total_charge_usd):
            self.calls.append((run_input, max_total_charge_usd))
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.01)
            self.running -= 1
            return SimpleNamespace(status=self.status, status_message="boom", default_dataset_id=run_input["source_id"])
        return SimpleNamespace(call=call)

    def dataset(self, dataset_id):
        async def iterate_items():
            for item in self.items_by_url.get(dataset_id, []):
                yield item
        return SimpleNamespace(iterate_items=iterate_items)


def test_apify_chunker_concurrency_and_budget(monkeypatch):
    pages = [Page(url=f"https://example.com/p{i}", markdown="# T\n\nx") for i in range(8)]
    fake = FakeApify({p.url: [{**RECORDED[0], "source": p.url}] * RECORDED[0]["chunk_total"] for p in pages})
    monkeypatch.setenv("APIFY_TOKEN", "test")
    monkeypatch.setattr("apify_client.ApifyClientAsync", lambda token: fake)

    chunks = chunk_with_apify(pages, concurrency=3)
    assert len(fake.calls) == 8 and fake.peak <= 3
    assert all(str(budget) == "0.50" for _, budget in fake.calls)
    assert len(chunks) == 8 * RECORDED[0]["chunk_total"]
    assert chunks[0]["text"] == RECORDED[0]["embedding_text"] and chunks[0]["section"] == RECORDED[0]["header_path"]


def test_apify_chunker_skips_failed_and_partial_runs(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "test")
    failed = FakeApify({PAGES[0].url: RECORDED}, status="FAILED")
    monkeypatch.setattr("apify_client.ApifyClientAsync", lambda token: failed)
    assert chunk_with_apify(PAGES[:1]) == []

    partial = FakeApify({PAGES[0].url: RECORDED[:1]})  # 1 of 2 chunks: charge cap reached
    monkeypatch.setattr("apify_client.ApifyClientAsync", lambda token: partial)
    assert chunk_with_apify(PAGES[:1]) == []


def test_dimension_mismatch_is_reported():
    client = QdrantClient(location=":memory:")
    client.create_collection("t", vectors_config={"dense": models.VectorParams(size=1536, distance=models.Distance.COSINE)})
    with pytest.raises(SystemExit, match="1536-dim"):
        upsert_chunks(client, chunk_local(PAGES), name="t")

