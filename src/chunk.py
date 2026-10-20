"""Step 2: split each page's Markdown into chunks.

Two interchangeable chunkers return the same record shape:

    {"id", "text", "content", "url", "section", "chunk_index", "document_hash"}

- ``chunk_local``: LangChain's recursive splitter, run on your machine, free.
- ``chunk_with_apify``: the Markdown RAG Chunker Actor on Apify. Header-aware
  (a chunk never crosses a section boundary), the section path is prepended to
  the embedded text, and ``id`` is stable across re-crawls so upserts replace
  points instead of duplicating them.
"""

import asyncio
import hashlib
import logging
import os
from collections.abc import Iterable
from decimal import Decimal

from .crawl import Page

ACTOR_ID = "codepoetry/markdown-rag-chunker"
log = logging.getLogger(__name__)


def document_hash(markdown: str) -> str:
    """Same fingerprint the Actor returns as ``document_hash`` for ``markdown_text`` input.

    Comparing it with the hash stored in Qdrant lets ingestion skip unchanged
    pages before paying for chunking or embeddings.
    """
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:16]


def chunk_local(pages: Iterable[Page], max_tokens: int = 512, overlap: int = 50) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # from_tiktoken_encoder measures chunk_size in tokens. The plain
    # constructor measures characters (512 characters is ~130 English tokens).
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = []
    for page in pages:
        doc_hash = document_hash(page.markdown)
        for i, text in enumerate(splitter.split_text(page.markdown)):
            chunks.append({
                # Positional ID: an edit near the top of a page shifts every
                # later chunk, so changed pages are fully re-embedded.
                "id": f"{page.url}#{i}",
                "text": text,
                "content": text,
                "url": page.url,
                "section": "",
                "chunk_index": i,
                "document_hash": doc_hash,
            })
    return chunks


def _to_chunk(item: dict) -> dict:
    return {
        "id": item["chunk_id"],
        "text": item["embedding_text"],   # section path + content: embed this
        "content": item["content"],       # show this in results
        "url": item["source"],
        "section": item["header_path"],
        "chunk_index": item["chunk_index"],
        "document_hash": item["document_hash"],
    }


async def _chunk_page(client, page: Page, max_tokens: int, max_charge_usd: Decimal) -> list[dict]:
    # One run per document at the default 1 GB memory (one $0.001 start event).
    # max_total_charge_usd caps what a single unexpectedly huge page can cost.
    run = await client.actor(ACTOR_ID).call(
        run_input={"markdown_text": page.markdown, "source_id": page.url, "max_chunk_tokens": max_tokens},
        max_total_charge_usd=max_charge_usd,
    )
    if run is None or run.status != "SUCCEEDED":
        log.warning("Chunking failed for %s: %s", page.url, run.status_message if run else "run did not start")
        return []
    items = [item async for item in client.dataset(run.default_dataset_id).iterate_items()]
    if items and len(items) < items[0]["chunk_total"]:
        # The budget cap stopped delivery. Indexing a partial page would later
        # look "unchanged" and never be completed, so skip it this time.
        log.warning("Skipped %s: %d of %d chunks within the $%s cap; raise --max-charge-per-page",
                    page.url, len(items), items[0]["chunk_total"], max_charge_usd)
        return []
    return [_to_chunk(item) for item in items]


def chunk_with_apify(
    pages: Iterable[Page],
    max_tokens: int = 512,
    concurrency: int = 4,
    max_charge_usd: Decimal = Decimal("0.50"),
) -> list[dict]:
    from apify_client import ApifyClientAsync

    async def run_all() -> list[dict]:
        client = ApifyClientAsync(os.environ["APIFY_TOKEN"])
        limit = asyncio.Semaphore(concurrency)

        async def bounded(page: Page) -> list[dict]:
            async with limit:
                return await _chunk_page(client, page, max_tokens, max_charge_usd)

        results = await asyncio.gather(*(bounded(page) for page in pages))
        return [chunk for page_chunks in results for chunk in page_chunks]

    return asyncio.run(run_all())
