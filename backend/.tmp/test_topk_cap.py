import os
import tempfile

tempfile.tempdir = r"d:\my-projects\nlp\HybridRAG\backend\.tmp"
os.environ["USE_RERANKER"] = "false"

import asyncio
from src.api.routers import chat, search
from src.config.settings import settings


class BM25:
    def __init__(self):
        self.k = None

    async def search(self, query, top_k):
        self.k = top_k
        return [{"content": "ok"}]


class VEC:
    def __init__(self):
        self.k = None

    async def search(self, query, top_k):
        self.k = top_k
        return [{"content": "ok"}]


class Dummy:
    def __init__(self):
        self.bm25 = BM25()
        self.vector = VEC()

    async def search(self, **kwargs):
        return [{"content": "ok"}]


async def main():
    dummy = Dummy()
    chat.get_hybrid_searcher = lambda: dummy
    search.get_hybrid_searcher = lambda: dummy

    settings.ELASTIC_SEARCH_K = 9
    settings.VECTOR_SEARCH_K = 8
    settings.SINGLE_MODE_SEARCH_MAX_K = 3

    await chat._retrieve_docs("q", "keyword")
    keyword_k_chat = dummy.bm25.k
    await chat._retrieve_docs("q", "semantic")
    semantic_k_chat = dummy.vector.k

    kw = await search.keyword_search(search.SearchRequest(query="test"), None)
    vc = await search.vector_search(search.SearchRequest(query="test"), None)

    print(f"chat-keyword-k={keyword_k_chat}")
    print(f"chat-semantic-k={semantic_k_chat}")
    print(f"api-keyword-applied={kw.config.get('APPLIED_TOP_K')}")
    print(f"api-vector-applied={vc.config.get('APPLIED_TOP_K')}")


if __name__ == '__main__':
    asyncio.run(main())
