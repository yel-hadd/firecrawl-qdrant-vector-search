"""Crawl a site, chunk it, and index it in Qdrant.

    python ingest.py https://docs.qdrant.tech --limit 50 --chunker apify
"""

import argparse

from dotenv import load_dotenv

from src.chunk import chunk_local, chunk_with_apify
from src.crawl import crawl_site
from src.index import delete_stale, get_client, upsert_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--limit", type=int, default=50, help="max pages to crawl (1 Firecrawl credit each)")
    parser.add_argument("--include", action="append", help="only crawl paths matching this pattern (repeatable)")
    parser.add_argument("--chunker", choices=["apify", "local"], default="apify")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    load_dotenv()
    pages = crawl_site(args.url, limit=args.limit, include_paths=args.include)
    print(f"Crawled {len(pages)} pages")

    if args.chunker == "apify":
        chunks = chunk_with_apify(pages, max_tokens=args.max_tokens)
    else:
        chunks = chunk_local(pages, max_tokens=args.max_tokens)
    print(f"Created {len(chunks)} chunks")

    client = get_client()
    print(f"Upserted {upsert_chunks(client, chunks)} points")
    delete_stale(client, chunks)


if __name__ == "__main__":
    main()
