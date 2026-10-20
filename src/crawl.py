"""Step 1: crawl a website to clean Markdown with Firecrawl."""

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from firecrawl import Firecrawl, ScrapeOptions

log = logging.getLogger(__name__)


@dataclass
class Page:
    """One document to index.

    ``url`` is the document's identity: chunk IDs, change detection and stale
    cleanup are keyed on it, so it must be stable and unique per document.
    For documents that are not web pages, use another stable ID such as a
    file path (``docs/handbook.pdf``) or a CMS record ID.
    """

    url: str
    markdown: str


def page_url(doc) -> str | None:
    """The URL that identifies a crawled document, or ``None``.

    ``source_url`` is the URL Firecrawl requested and ``url`` the one it loaded.
    ``og_url`` is deliberately ignored: many sites set ``og:url`` to the
    homepage on every page, which would give different pages one identity.
    """
    meta = doc.metadata
    return (getattr(meta, "source_url", None) or getattr(meta, "url", None)) if meta else None


def pages_from_documents(docs: Iterable) -> list[Page]:
    """One Page per unique URL with non-empty Markdown; the last copy of a URL wins."""
    pages: dict[str, Page] = {}
    missing_url = 0
    for doc in docs:
        if not (doc.markdown and doc.markdown.strip()):
            continue
        url = page_url(doc)
        if not url:
            missing_url += 1
            continue
        pages[url] = Page(url=url, markdown=doc.markdown)
    if missing_url:
        # Indexing them under a shared or made-up ID would let one page
        # overwrite or delete another's chunks, so they are left out.
        log.warning("Skipped %d crawled page(s) without a URL: they can't be cited or kept in sync.", missing_url)
    return list(pages.values())


def crawl_site(url: str, limit: int = 50, include_paths: list[str] | None = None) -> list[Page]:
    """Crawl ``url`` and return its pages as clean Markdown.

    ``only_main_content`` strips navigation, footers and cookie banners before
    the Markdown is returned, which keeps boilerplate out of your embeddings.
    """
    firecrawl = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])
    job = firecrawl.crawl(
        url,
        limit=limit,
        include_paths=include_paths,
        scrape_options=ScrapeOptions(formats=["markdown"], only_main_content=True),
    )
    return pages_from_documents(job.data or [])
