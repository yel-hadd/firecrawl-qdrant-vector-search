"""Step 5: semantic and hybrid (dense + BM25) search."""

from qdrant_client import QdrantClient, models

from .embed import embed_dense, embed_sparse_query
from .index import COLLECTION


def _results(points) -> list[dict]:
    return [
        {
            "score": p.score,
            "url": p.payload["url"],
            "section": p.payload.get("section", ""),
            "text": p.payload["content"][:300],
        }
        for p in points
    ]


def semantic_search(client: QdrantClient, query: str, limit: int = 5, name: str = COLLECTION) -> list[dict]:
    response = client.query_points(
        collection_name=name,
        query=embed_dense([query])[0],
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _results(response.points)


def hybrid_search(client: QdrantClient, query: str, limit: int = 5, name: str = COLLECTION) -> list[dict]:
    indices, values = embed_sparse_query(query)
    response = client.query_points(
        collection_name=name,
        prefetch=[
            models.Prefetch(query=embed_dense([query])[0], using="dense", limit=20),
            models.Prefetch(query=models.SparseVector(indices=indices, values=values), using="sparse", limit=20),
        ],
        # Reciprocal Rank Fusion merges both ranked lists without score tuning.
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _results(response.points)
