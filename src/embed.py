"""Step 3: dense and sparse embeddings.

Dense vectors use OpenAI ``text-embedding-3-small`` (1536 dims) when
``OPENAI_API_KEY`` is set, otherwise a local FastEmbed model (384 dims, CPU,
no API key). Sparse BM25 vectors always come from FastEmbed and power the
keyword half of hybrid search.
"""

import os
from functools import cache

OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


def use_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def dense_size() -> int:
    return 1536 if use_openai() else 384


@cache
def _local_dense():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=LOCAL_MODEL)


@cache
def _sparse():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_dense(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    if not use_openai():
        return [vector.tolist() for vector in _local_dense().embed(texts)]

    from openai import OpenAI

    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=OPENAI_MODEL, input=texts[start:start + batch_size])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    return [(e.indices.tolist(), e.values.tolist()) for e in _sparse().embed(texts)]


def embed_sparse_query(text: str) -> tuple[list[int], list[float]]:
    e = next(iter(_sparse().query_embed(text)))
    return e.indices.tolist(), e.values.tolist()
