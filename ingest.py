"""Crawl a site, chunk it, and index it in Qdrant.

    python ingest.py https://docs.qdrant.tech --limit 50 --chunker apify

Re-runs only chunk and embed pages whose content changed since the last run.
"""

import argparse
import logging

from dotenv import load_dotenv

from src.chunk import chunk_local, chunk_with_apify, document_hash
from src.crawl import crawl_site
from src.index import delete_stale, get_client, indexed_hashes, upsert_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--limit", type=int, default=50, help="max pages to crawl (1 Firecrawl credit each)")
    parser.add_argument("--include", action="append", help="only crawl URL paths matching this regex (repeatable)")
    parser.add_argument("--chunker", choices=["apify", "local"], default="apify")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4, help="parallel Apify runs (apify chunker)")
    parser.add_argument("--max-chunks-per-page", type=int, default=600,
                        help="skip pages that would produce more chunks; they cost only the $0.001 run fee (apify chunker)")
    parser.add_argument("--force", action="store_true", help="re-chunk and re-embed unchanged pages too")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    load_dotenv()
    pages = crawl_site(args.url, limit=args.limit, include_paths=args.include)
    print(f"Crawled {len(pages)} pages")

    client = get_client()
    if not args.force:
        # Skip pages already indexed with the same content: no chunking runs and no embeddings.
        known = indexed_hashes(client, [p.url for p in pages])
        pages = [p for p in pages if known.get(p.url) != {document_hash(p.markdown)}]
    print(f"{len(pages)} new or changed pages to index")

    if args.chunker == "apify":
        chunks = chunk_with_apify(pages, max_tokens=args.max_tokens, concurrency=args.concurrency,
                                  max_chunks=args.max_chunks_per_page)
    else:
        chunks = chunk_local(pages, max_tokens=args.max_tokens)
    print(f"Created {len(chunks)} chunks")

    print(f"Upserted {upsert_chunks(client, chunks)} points")
    delete_stale(client, chunks)


if __name__ == "__main__":
    main()
