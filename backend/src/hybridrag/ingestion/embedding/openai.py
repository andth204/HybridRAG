from typing import List, Union
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
        response = self.sync_client.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def embed_text(self, text: str) -> np.ndarray:
        response = await self.async_client.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    async def embed(self, text_or_texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(text_or_texts, str):
            return await self.embed_text(text_or_texts)
        elif isinstance(text_or_texts, list):
            return [await self.embed_text(text) for text in text_or_texts]
        else:
            raise ValueError(f"Input must be str or List[str], got {type(text_or_texts)}")

    def get_dimension(self) -> int:
        return self.dimension


embedder = OpenAIEmbedder()
__all__ = ["OpenAIEmbedder", "embedder"]