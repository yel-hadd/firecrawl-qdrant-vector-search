"""Step 1: crawl a website to clean Markdown with Firecrawl."""

import os
from dataclasses import dataclass

from firecrawl import Firecrawl, ScrapeOptions


@dataclass
class Page:
    url: str
    markdown: str


def crawl_site(url: str, limit: int = 50, include_paths: list[str] | None = None) -> list[Page]:
    """Crawl ``url`` and return one Page per crawled document with non-empty Markdown.

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
    pages = []
    for doc in job.data:
        source = (doc.metadata.source_url if doc.metadata else None) or url
        if doc.markdown and doc.markdown.strip():
            pages.append(Page(url=source, markdown=doc.markdown))
    return pages
