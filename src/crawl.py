"""Step 1: crawl a website to clean Markdown with Firecrawl."""

import os
from dataclasses import dataclass

from firecrawl import Firecrawl, ScrapeOptions


@dataclass
class Page:
    url: str
    markdown: str


def crawl_site(url: str, limit: int = 50, include_paths: list[str] | None = None) -> list[Page]:
    """Crawl ``url`` and return one Page per unique URL with non-empty Markdown.

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
    pages: dict[str, Page] = {}
    for doc in job.data or []:
        source = doc.metadata.source_url if doc.metadata else None
        # A page without its own URL can't be cited or kept in sync: skip it.
        if source and doc.markdown and doc.markdown.strip():
            pages[source] = Page(url=source, markdown=doc.markdown)
    return list(pages.values())
