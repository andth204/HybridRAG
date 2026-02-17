from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    
    # OpenAI & LLM
    OPENAI_API_KEY: str = ""
    GENERATE_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    TEMPERATURE_MAIN: float = 0.6
    TEMPERATURE_CHITCHAT: float = 0.8
    MAX_GEN_MAIN: int = 500
    MAX_GEN_CHITCHAT: int = 200

    # Reranker
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "jinaai/jina-reranker-v2-base-multilingual"

    # Chunking
    FIXED_CHUNK_SIZE: int = 1024
    FIXED_CHUNK_OVERLAP: int = 180
    
    # Retrieval
    VECTOR_SEARCH_K: int = 7
    ELASTIC_SEARCH_K: int = 7
    RRF_K: int = 60  
    FUSION_K: int = 10
    RERANK_TOP_K: int = 3

    # Query Rewriting
    MAX_HISTORY_TOKENS_REWRITE: int = 300
    TEMPERATURE_REWRITER: float = 0.3
    K_REWRITE: int = 10

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "utehy"
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = ""
    MINIO_ROOT_PASSWORD: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = ""

    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "samples"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"  
    ROUTER_EMBEDDINGS_DIR: Path = VECTOR_STORE_DIR / "router_embeddings"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        case_sensitive = True
    
    def model_post_init(self, __context):
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True) 
        self.ROUTER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)  

settings = Settings()