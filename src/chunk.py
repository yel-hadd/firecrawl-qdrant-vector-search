"""Step 2: split each page's Markdown into chunks.

Two interchangeable chunkers return the same record shape:

    {"id", "text", "content", "url", "section", "chunk_index", "document_hash"}

- ``chunk_local``: LangChain's recursive splitter, run on your machine, free.
- ``chunk_with_apify``: the Markdown RAG Chunker Actor on Apify. Header-aware
  (a chunk never crosses a section boundary), the section path is prepended to
  the embedded text, and ``id`` is stable across re-crawls so upserts replace
  points instead of duplicating them.
"""

import hashlib
import os
from collections.abc import Iterable

from .crawl import Page

ACTOR_ID = "codepoetry/markdown-rag-chunker"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
        doc_hash = _hash(page.markdown)
        for i, text in enumerate(splitter.split_text(page.markdown)):
            chunks.append({
                # Positional ID: re-crawls overwrite by position, so an edit
                # near the top of a page shifts every later chunk.
                "id": f"{page.url}#{i}",
                "text": text,
                "content": text,
                "url": page.url,
                "section": "",
                "chunk_index": i,
                "document_hash": doc_hash,
            })
    return chunks


def chunk_with_apify(pages: Iterable[Page], max_tokens: int = 512, skip_if_unchanged: bool = False) -> list[dict]:
    from apify_client import ApifyClient

    client = ApifyClient(os.environ["APIFY_TOKEN"])
    chunks = []
    for page in pages:
        # One run per document. source_id scopes chunk IDs to this URL.
        run = client.actor(ACTOR_ID).call(run_input={
            "markdown_text": page.markdown,
            "source_id": page.url,
            "max_chunk_tokens": max_tokens,
            "skip_if_unchanged": skip_if_unchanged,
        })
        if run is None:
            raise RuntimeError(f"Actor run for {page.url} did not start")
        for item in client.dataset(run.default_dataset_id).iterate_items():
            chunks.append({
                "id": item["chunk_id"],
                "text": item["embedding_text"],   # section path + content: embed this
                "content": item["content"],       # show this in results
                "url": item["source"],
                "section": item["header_path"],
                "chunk_index": item["chunk_index"],
                "document_hash": item["document_hash"],
            })
    return chunks
