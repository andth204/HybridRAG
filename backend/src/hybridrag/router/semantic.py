import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .route import Route
from src.hybridrag.ingestion.embedding import OpenAIEmbedder
import faiss
from cachetools import TTLCache

class SemanticRouter:
    def __init__(
        self, 
        routes: List[Route],
        embedder: Optional[OpenAIEmbedder] = None,
        embeddings_dir: Path = None,
        max_cache_size: int = 1000,
        ttl_seconds: int = 86400
    ):
        self.routes = routes
        self.embedder = embedder or OpenAIEmbedder()
        self.embeddings_dir = embeddings_dir
        self.routes_embeddings: Dict[str, List[np.ndarray]] = {}
        self.query_cache = TTLCache(maxsize=max_cache_size, ttl=ttl_seconds)
        self.embedding_cache = TTLCache(maxsize=max_cache_size, ttl=ttl_seconds)
        self._load_embeddings()
        self._build_faiss_index()

    def _get_embeddings_path(self, route_name: str) -> Path:
        return self.embeddings_dir / f"{route_name}_embeddings.pkl"
    
    def _load_embeddings(self):
        for route in self.routes:
            embeddings_path = self._get_embeddings_path(route.name)
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            self.routes_embeddings[route.name] = data['embeddings']

    def _build_faiss_index(self):
        all_embeddings = []
        self.index_to_route = []
        for route in self.routes:
            embs = self.routes_embeddings[route.name]
            if not embs:
                continue
            embs = np.array(embs, dtype=np.float32)
            faiss.normalize_L2(embs)
            all_embeddings.append(embs)
            self.index_to_route.extend([route.name] * len(embs))

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            dim = all_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(all_embeddings)
        else:
            self.faiss_index = None

    async def _get_cached_or_embed(self, query: str) -> np.ndarray:
        if query in self.embedding_cache:
            return self.embedding_cache[query]
        embedding = await self.embedder.embed(query)
        embedding_array = np.array(embedding, dtype=np.float32)
        self.embedding_cache[query] = embedding_array
        return embedding_array

    async def guide(self, query: str) -> Tuple[float, str]:
        if query in self.query_cache:
            return self.query_cache[query]

        if self.faiss_index is None:
            result = (0.0, "chitchat")
            self.query_cache[query] = result
            return result

        query_embedding = await self._get_cached_or_embed(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        _, indices = self.faiss_index.search(query_embedding, k=1)
        best_idx = int(indices[0][0])
        best_route = self.index_to_route[best_idx]

        scores, _ = self.faiss_index.search(query_embedding, k=1)
        best_score = float(scores[0][0])

        result = (best_score, best_route)
        self.query_cache[query] = result
        return result

    async def guide_with_all_scores(self, query: str) -> List[Tuple[float, str]]:
        if self.faiss_index is None:
            return [(0.0, "chitchat")]
        query_embedding = await self._get_cached_or_embed(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        k = min(self.faiss_index.ntotal, 10)
        scores, indices = self.faiss_index.search(query_embedding, k=k)
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            route = self.index_to_route[idx]
            results.append((score, route))
        return sorted(results, reverse=True, key=lambda x: x[0])

    def get_routes(self) -> List[Route]:
        return self.routes

__all__ = ["SemanticRouter"]