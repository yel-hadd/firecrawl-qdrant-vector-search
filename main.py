"""Search API over the indexed site.

    uvicorn main:api --reload
    curl "http://localhost:8000/search?q=configure+authentication&mode=hybrid"
"""

from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.index import COLLECTION, get_client
from src.search import hybrid_search, semantic_search

load_dotenv()
api = FastAPI(title="Website Search API")
client = get_client()


class SearchResult(BaseModel):
    score: float
    url: str
    section: str
    text: str


@api.get("/search", response_model=list[SearchResult])
def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20),
    mode: Literal["semantic", "hybrid"] = "hybrid",
):
    try:
        ready = client.collection_exists(COLLECTION)
    except Exception as exc:
        raise HTTPException(503, f"Qdrant is unreachable: {exc}") from exc
    if not ready:
        raise HTTPException(503, f"Collection '{COLLECTION}' not found. Run ingest.py first.")
    if mode == "semantic":
        return semantic_search(client, q, limit)
    return hybrid_search(client, q, limit)
