# Firecrawl + Qdrant: semantic and hybrid search over any website

Crawl a website to Markdown with [Firecrawl](https://firecrawl.link/yassine-el-haddad), split it into header-aware RAG chunks with [Markdown RAG Chunker](https://apify.com/codepoetry/markdown-rag-chunker?fpr=use-apify), embed them, and serve **hybrid search** (dense vectors + BM25, fused with Reciprocal Rank Fusion) from [Qdrant](https://qdrant.tech) behind a FastAPI endpoint.

Companion code for the step-by-step tutorial **[Firecrawl + Qdrant: Build a Vector Search Engine for Any Website](https://use-apify.com/blog/firecrawl-qdrant-vector-search)** on use-apify.com, which explains every design choice below.

```
Firecrawl /crawl ──▶ Markdown per page ──▶ chunks (512 tokens, one section each)
                                              │
                    FastAPI /search ◀── Qdrant (dense + sparse) ◀── embeddings
```

## What you get

- **Crawl**: Firecrawl Python SDK v4 (`firecrawl.crawl`) with `only_main_content`, so navigation and cookie banners never reach your embeddings.
- **Chunk**, two interchangeable options:
  - `--chunker apify` (default): [Markdown RAG Chunker](https://apify.com/codepoetry/markdown-rag-chunker?fpr=use-apify) splits on headings first, caps sections at 512 real tokens, prepends the section path to the embedded text, and returns a **stable `chunk_id`**, so re-crawls update points instead of duplicating them.
  - `--chunker local`: LangChain's `RecursiveCharacterTextSplitter.from_tiktoken_encoder`. Free, runs locally, positional IDs.
- **Embed**: OpenAI `text-embedding-3-small` when `OPENAI_API_KEY` is set, otherwise a local FastEmbed model (`BAAI/bge-small-en-v1.5`, CPU, no key). BM25 sparse vectors from FastEmbed.
- **Index**: Qdrant named dense + sparse vectors with the IDF modifier, deterministic UUID point IDs, and stale-chunk cleanup by `document_hash`.
- **Search**: `query_points` with dense, sparse and RRF prefetch (the current Qdrant query API; `search` and `recreate_collection` are gone or deprecated in `qdrant-client` 1.19).

## Quick start

```bash
git clone https://github.com/yel-hadd/firecrawl-qdrant-vector-search
cd firecrawl-qdrant-vector-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add FIRECRAWL_API_KEY, APIFY_TOKEN, optional OPENAI_API_KEY

docker run -p 6333:6333 qdrant/qdrant

python ingest.py https://docs.qdrant.tech --limit 50
uvicorn main:api --reload
curl "http://localhost:8000/search?q=configure+authentication&mode=hybrid"
```

Run the offline test (no API keys, in-memory Qdrant, local embeddings):

```bash
pytest
```

## Which chunker should I use?

| | `--chunker apify` (Markdown RAG Chunker) | `--chunker local` (LangChain) |
|---|---|---|
| Chunk boundaries | Never crosses a heading | Prefers headings, can cross them |
| Section context | `header_path` in payload and in the embedded text | None |
| Chunk IDs | Stable across re-crawls (content + source) | Position in page |
| Token count | Exact, tiktoken `cl100k_base` | Exact, tiktoken `cl100k_base` |
| Unchanged pages on a schedule | `skip_if_unchanged`: no chunks, $0.001 run fee | Re-chunked every time |
| Cost | $0.80 per 1,000 chunks + $0.001 per run (Apify free plan) | Free |
| Runs | One Apify run per page | Locally |

Pick `local` for a one-off index. Pick `apify` when you re-crawl on a schedule and want unchanged sections to keep their points, or when a page's section path matters for retrieval (API references, docs with repeated subheadings like "Parameters").

## Costs for a 50-page site

| Item | Cost |
|---|---|
| Firecrawl crawl | 50 credits ([free plan: 1,000 credits/month](https://firecrawl.link/yassine-el-haddad)) |
| Markdown RAG Chunker (~190 chunks) | ~$0.20 (50 runs × $0.001 + 190 × $0.0008) |
| OpenAI `text-embedding-3-small` | well under $0.01 |
| Qdrant | free (Docker or Qdrant Cloud free tier) |

Prices as of October 2026. Check the [Firecrawl](https://firecrawl.link/yassine-el-haddad) and [Apify Actor pricing](https://apify.com/codepoetry/markdown-rag-chunker/pricing?fpr=use-apify) pages before budgeting.

## Project layout

```
ingest.py          # CLI: crawl → chunk → embed → upsert → delete stale
main.py            # FastAPI /search endpoint (semantic | hybrid)
src/crawl.py       # Firecrawl crawl to Markdown
src/chunk.py       # Markdown RAG Chunker (Apify) or LangChain splitter
src/embed.py       # OpenAI or FastEmbed dense + BM25 sparse
src/index.py       # Qdrant collection, UUID point IDs, stale cleanup
src/search.py      # semantic and RRF hybrid search
tests/             # offline end-to-end test
```

## Crawl JavaScript-heavy or protected sites

Firecrawl handles most documentation and marketing sites. For sites behind logins, aggressive bot protection, or when you want crawling, chunking and scheduling on one platform, swap `crawl_site` for Apify's [Website Content Crawler](https://apify.com/apify/website-content-crawler?fpr=use-apify): it returns the same Markdown per page, and the rest of this pipeline is unchanged. Comparison: [Apify vs Firecrawl](https://use-apify.com/docs/apify-vs-the-world/apify-vs-firecrawl).

## FAQ

**How do I build vector search over a website?**
Crawl it to Markdown (Firecrawl or Website Content Crawler), split the Markdown into chunks of a few hundred tokens, embed each chunk, and upsert the vectors to a vector database such as Qdrant with the source URL in the payload. At query time, embed the question and return the nearest chunks. This repo does all five steps.

**What chunk size should I use for RAG?**
Start at 512 tokens. Chroma's [chunking evaluation](https://www.trychroma.com/research/evaluating-chunking) found recursive splitting at a few hundred tokens retrieves best, and NVIDIA's [2025 chunking study](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) recommends 256 to 512 tokens for factoid queries. Measure in tokens, not characters.

**Why hybrid search instead of only vectors?**
Dense vectors miss exact strings such as error codes, product names and version numbers. BM25 catches them. RRF merges both ranked lists without score normalisation.

**How do I keep the index in sync when pages change?**
Upsert by a stable chunk ID, then delete points whose `url` matches the page and whose `document_hash` differs from the new one (`delete_stale` in `src/index.py`).

## More guides

- [Firecrawl + Qdrant tutorial](https://use-apify.com/blog/firecrawl-qdrant-vector-search): the full walkthrough for this repo
- [Firecrawl for RAG: build a knowledge base from any website](https://use-apify.com/blog/firecrawl-rag-knowledge-base)
- [Scraping technical documentation to Markdown for Claude](https://use-apify.com/blog/scrape-docs-to-markdown-claude)
- [Web data for AI and RAG with Apify](https://use-apify.com/docs/apify-use-cases/data-for-ai-rag)

Some links above are affiliate links: if you sign up through them, use-apify.com may earn a commission at no cost to you.

## License

MIT
