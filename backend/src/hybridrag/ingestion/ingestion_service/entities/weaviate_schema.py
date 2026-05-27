"""Weaviate schema for the DocChunk class. Idempotent collection creation."""
from __future__ import annotations

import logging
from typing import Any

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)

from src.config.settings import settings

log = logging.getLogger(__name__)


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    """Create the DocChunk collection if missing. No-op if already there.

    The collection is configured for:
      - External embeddings (vectorizer = none); vectors are supplied by us at insert time.
      - HNSW vector index tuned via WEAVIATE_HNSW_* settings.
      - Inverted index tuned for BM25 with WEAVIATE_BM25_K1 / WEAVIATE_BM25_B.
      - Word tokenization on `content` so BM25 works on full text.
      - All metadata properties are stored with skip_vectorization=True so they do
        not contaminate the embedding space (vectors come from an external model).

    Parameters
    ----------
    client : weaviate.WeaviateClient
        A connected v4 Weaviate client.
    """
    name: str = settings.WEAVIATE_CLASS_NAME
    if client.collections.exists(name):
        log.info("Weaviate collection %s already exists", name)
        return

    log.info("Creating Weaviate collection %s", name)
    client.collections.create(
        name=name,
        description="Hierarchical document chunks for HybridRAG",
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef=settings.WEAVIATE_HNSW_EF,
            ef_construction=settings.WEAVIATE_HNSW_EF_CONSTRUCTION,
            max_connections=settings.WEAVIATE_HNSW_MAX_CONNECTIONS,
        ),
        inverted_index_config=Configure.inverted_index(
            bm25_b=settings.WEAVIATE_BM25_B,
            bm25_k1=settings.WEAVIATE_BM25_K1,
        ),
        properties=[
            Property(name="chunk_id",    data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="parent_id",   data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="file_id",     data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="key",         data_type=DataType.TEXT, skip_vectorization=True),
            Property(
                name="content",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(name="section",     data_type=DataType.TEXT),
            Property(name="header_path", data_type=DataType.TEXT_ARRAY),
            Property(name="chunk_level", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="is_table",    data_type=DataType.BOOL, skip_vectorization=True),
            Property(name="campus",      data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="doc_type",    data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="faculty",     data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="major",       data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="year",        data_type=DataType.INT,  skip_vectorization=True),
        ],
    )
    log.info("Weaviate collection %s created", name)
