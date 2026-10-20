"""Step 4: store chunks in Qdrant with dense + sparse vectors."""

import os
import uuid

from qdrant_client import QdrantClient, models

from .embed import dense_size, embed_dense, embed_sparse

COLLECTION = os.environ.get("QDRANT_COLLECTION", "website_search")


def get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))


def ensure_collection(client: QdrantClient, name: str = COLLECTION) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=dense_size(), distance=models.Distance.COSINE)},
        # IDF lets Qdrant weight BM25 terms by how rare they are in the collection.
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(name, "url", models.PayloadSchemaType.KEYWORD)


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be unsigned integers or UUIDs, so derive a UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client: QdrantClient, chunks: list[dict], name: str = COLLECTION, batch_size: int = 100) -> int:
    ensure_collection(client, name)
    texts = [c["text"] for c in chunks]
    dense = embed_dense(texts)
    sparse = embed_sparse(texts)
    points = [
        models.PointStruct(
            id=point_id(c["id"]),
            vector={"dense": d, "sparse": models.SparseVector(indices=s[0], values=s[1])},
            payload={k: c[k] for k in ("content", "url", "section", "chunk_index", "document_hash")},
        )
        for c, d, s in zip(chunks, dense, sparse)
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=name, points=points[start:start + batch_size])
    return len(points)


def delete_stale(client: QdrantClient, chunks: list[dict], name: str = COLLECTION) -> None:
    """Remove points left over from older versions of the re-crawled pages."""
    current = {c["url"]: c["document_hash"] for c in chunks}
    for url, doc_hash in current.items():
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))],
                must_not=[models.FieldCondition(key="document_hash", match=models.MatchValue(value=doc_hash))],
            )),
        )
