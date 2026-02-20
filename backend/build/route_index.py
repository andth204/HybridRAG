import asyncio
import numpy as np
import pickle
from src.hybridrag.router.samples import ROUTES
from src.hybridrag.ingestion.embedding.openai import embedder
from src.config.settings import settings


async def build_and_save(route, embedder):
    output_dir = settings.ROUTER_EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = await embedder.embed(route.samples)
    data = {
        "route_name": route.name,
        "embeddings": embeddings,
        "embedding_dim": embeddings.shape[1],
        "sample_count": embeddings.shape[0],
    }
    output_file = output_dir / f"{route.name}_embeddings.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Done embedding for router: {route.name}!!!")

async def main():
    for route in ROUTES:
        await build_and_save(route, embedder)

if __name__ == "__main__":
    asyncio.run(main())