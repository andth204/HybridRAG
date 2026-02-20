from typing import List, Union
import asyncio
import numpy as np
from openai import OpenAI, AsyncOpenAI
from src.config.settings import settings


class OpenAIEmbedder:
    def __init__(self, api_key: str = None, model: str = None, dimension: int = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def embed_text_sync(self, text: str) -> np.ndarray:
        response = self.sync_client.embeddings.create(input=text, model=self.model, dimensions=self.dimension)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def embed_text(self, text: str) -> np.ndarray:
        response = await self.async_client.embeddings.create(input=text, model=self.model, dimensions=self.dimension)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def embed(self, text_or_texts: Union[str, List[str]], batch_size: int = 32, max_concurrency: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(text_or_texts, str):
            return await self.embed_text(text_or_texts)
        if not isinstance(text_or_texts, list):
            raise ValueError(
                f"Input must be str or List[str], got {type(text_or_texts)}"
            )
        semaphore = asyncio.Semaphore(max_concurrency)
        async def embed_batch(batch: List[str]) -> List[np.ndarray]:
            async with semaphore:
                response = await self.async_client.embeddings.create(input=batch, model=self.model, dimensions=self.dimension)
                return [
                    np.array(d.embedding, dtype=np.float32)
                    for d in response.data
                ]
        tasks = []
        for i in range(0, len(text_or_texts), batch_size):
            batch = text_or_texts[i:i + batch_size]
            tasks.append(embed_batch(batch))
        results = await asyncio.gather(*tasks)
        flattened = [item for sublist in results for item in sublist]
        return np.vstack(flattened)

    def get_dimension(self) -> int:
        return self.dimension


embedder = OpenAIEmbedder()

__all__ = ["OpenAIEmbedder", "embedder"]