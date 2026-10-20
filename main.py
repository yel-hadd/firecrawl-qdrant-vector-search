"""Search API over the indexed site.

    uvicorn main:api --reload
    curl "http://localhost:8000/search?q=configure+authentication&mode=hybrid"
"""

from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.index import get_client
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
    if mode == "semantic":
        return semantic_search(client, q, limit)
    return hybrid_search(client, q, limit)
