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
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)


def ensure_collection(client: QdrantClient, name: str = COLLECTION) -> None:
    size = dense_size()
    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors["dense"].size
        if existing != size:
            raise SystemExit(
                f"Collection '{name}' stores {existing}-dim vectors but the current embedding model "
                f"produces {size}. Set QDRANT_COLLECTION to a new name, or delete the collection."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
        # IDF lets Qdrant weight BM25 terms by how rare they are in the collection.
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(name, "url", models.PayloadSchemaType.KEYWORD)


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be unsigned integers or UUIDs, so derive a UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def indexed_hashes(client: QdrantClient, urls: list[str], name: str = COLLECTION) -> dict[str, set[str]]:
    """``document_hash`` values already stored per URL, used to skip unchanged pages."""
    if not urls or not client.collection_exists(name):
        return {}
    hashes: dict[str, set[str]] = {}
    offset = None
    url_filter = models.Filter(must=[models.FieldCondition(key="url", match=models.MatchAny(any=urls))])
    while True:
        points, offset = client.scroll(
            name, scroll_filter=url_filter, limit=1000, offset=offset,
            with_payload=["url", "document_hash"],
        )
        for p in points:
            hashes.setdefault(p.payload["url"], set()).add(p.payload["document_hash"])
        if offset is None:
            return hashes


def upsert_chunks(client: QdrantClient, chunks: list[dict], name: str = COLLECTION, batch_size: int = 100) -> int:
    ensure_collection(client, name)
    if not chunks:
        return 0
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
    """Remove older versions of the pages in ``chunks``. Other pages are left alone."""
    current = {c["url"]: c["document_hash"] for c in chunks}
    for url, doc_hash in current.items():
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))],
                must_not=[models.FieldCondition(key="document_hash", match=models.MatchValue(value=doc_hash))],
            )),
        )
