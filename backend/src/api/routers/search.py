from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from src.api.core.dependencies import AuthContext, get_auth_context
from src.api.core.runtime import get_hybrid_searcher
from src.config.settings import settings

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SearchResponse(BaseModel):
    mode: str
    query: str
    items: list[dict[str, Any]]
    config: dict[str, Any]


@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(
    payload: SearchRequest,
    _: AuthContext = Depends(get_auth_context),
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query must not be empty",
        )

    searcher = get_hybrid_searcher()
    single_mode_limit = max(1, int(getattr(settings, "SINGLE_MODE_SEARCH_MAX_K", 3)))
    keyword_k = min(int(settings.ELASTIC_SEARCH_K), single_mode_limit)
    items = await searcher.bm25.search(query=query, top_k=keyword_k)
    return SearchResponse(
        mode="keyword",
        query=query,
        items=items,
        config={
            "ELASTIC_SEARCH_K": settings.ELASTIC_SEARCH_K,
            "SINGLE_MODE_SEARCH_MAX_K": single_mode_limit,
            "APPLIED_TOP_K": keyword_k,
        },
    )


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    payload: SearchRequest,
    _: AuthContext = Depends(get_auth_context),
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query must not be empty",
        )

    searcher = get_hybrid_searcher()
    single_mode_limit = max(1, int(getattr(settings, "SINGLE_MODE_SEARCH_MAX_K", 3)))
    vector_k = min(int(settings.VECTOR_SEARCH_K), single_mode_limit)
    items = await searcher.vector.search(query=query, top_k=vector_k)
    return SearchResponse(
        mode="vector",
        query=query,
        items=items,
        config={
            "VECTOR_SEARCH_K": settings.VECTOR_SEARCH_K,
            "SINGLE_MODE_SEARCH_MAX_K": single_mode_limit,
            "APPLIED_TOP_K": vector_k,
        },
    )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    payload: SearchRequest,
    _: AuthContext = Depends(get_auth_context),
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query must not be empty",
        )

    searcher = get_hybrid_searcher()
    items = await searcher.search(
        query=query,
        vector_k=settings.VECTOR_SEARCH_K,
        bm25_k=settings.ELASTIC_SEARCH_K,
        fusion_top_n=settings.FUSION_K,
        use_reranker=settings.USE_RERANKER,
        rerank_top_k=settings.RERANK_TOP_K,
    )
    return SearchResponse(
        mode="hybrid",
        query=query,
        items=items,
        config={
            "VECTOR_SEARCH_K": settings.VECTOR_SEARCH_K,
            "ELASTIC_SEARCH_K": settings.ELASTIC_SEARCH_K,
            "FUSION_K": settings.FUSION_K,
            "USE_RERANKER": settings.USE_RERANKER,
            "RERANK_TOP_K": settings.RERANK_TOP_K,
            "RRF_K": settings.RRF_K,
        },
    )
