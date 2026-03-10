from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # -----< Auth >-----
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_TOKENINFO_URL: str = "https://oauth2.googleapis.com/tokeninfo"
    GOOGLE_REQUIRE_VERIFIED_EMAIL: bool = True
    GOOGLE_ALLOWED_HD: str = ""
    AUTH_ACCESS_TOKEN_TTL_MINUTES: int = 60
    AUTH_REFRESH_TOKEN_TTL_DAYS: int = 30
    AUTH_TOKEN_SCHEMA: str = "public"
    AUTH_TOKEN_TABLE: str = "auth_tokens"

    # -----< OpenAI & LLM >----- 
    OPENAI_API_KEY:       str   = ""
    GENERATE_MODEL:       str   = "gpt-4o-mini"
    EMBEDDING_MODEL:      str   = "text-embedding-3-small"
    EMBEDDING_DIMENSION:  int   = 1536
    TEMPERATURE_MAIN:     float = 0.6
    TEMPERATURE_CHITCHAT: float = 0.8
    MAX_GEN_MAIN:         int   = 500
    MAX_GEN_CHITCHAT:     int   = 100

    # -----< Chunking >-----
    FIXED_CHUNK_SIZE:    int = 1200
    FIXED_CHUNK_OVERLAP: int = 180
    
    # -----< Retrieval/ Reranker >-----
    VECTOR_SEARCH_K:  int = 7
    ELASTIC_SEARCH_K: int = 7
    RRF_K:            int = 60  
    FUSION_K:         int = 7
    RERANKER_MODEL:   str = "jinaai/jina-reranker-v2-base-multilingual"
    USE_RERANKER:    bool = True
    RERANK_TOP_K:     int = 3

    # -----< Query Rewriting >-----
    MAX_HISTORY_TOKENS_REWRITE: int = 250
    TEMPERATURE_REWRITER:     float = 0.2
    K_REWRITE:                  int = 8
    REWRITER_MODEL:             str = "gpt-4o-mini"
    REWRITER_TIMEOUT:         float = 2
    MAX_REWRITE_OUTPUT_TOKENS:  int = 100
    MAX_HISTORY_CHARS_REWRITE:  int = 1200
    REWRITER_CACHE_SIZE:        int = 512
    REWRITER_CACHE_TTL_SECONDS: int = 86400

    # -----< Kafka >-----
    KAFKA_BOOTSTRAP_SERVERS:   str = "localhost:9092"
    INDEXING_INPUT_TOPIC:      str = "minio-file-events"
    INDEXING_STATUS_TOPIC:     str = "minio-file-status"
    INDEXING_CONSUMER_GROUP:   str = "indexing-group-1"
    INDEXING_BATCH_SIZE:       int = 10
    INDEXING_BATCH_INTERVAL: float = 2.0

    # -----< PostgreSQL >-----
    POSTGRES_HOST:     str = "localhost"
    POSTGRES_PORT:     int = 5432
    POSTGRES_DB:       str = "utehy"
    POSTGRES_USER:     str = ""
    POSTGRES_PASSWORD: str = ""
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # -----< Index-state table (for idempotent + replace) >-----
    INDEX_STATE_SCHEMA: str = "public"
    INDEX_STATE_TABLE:  str = "file_index_state"

    # -----< MinIO >-----
    MINIO_ENDPOINT:      str = "http://localhost:9000"
    MINIO_ROOT_USER:     str = ""
    MINIO_ROOT_PASSWORD: str = ""
    MINIO_ACCESS_KEY:    str = ""
    MINIO_SECRET_KEY:    str = ""
    MINIO_SECURE:       bool = False
    MINIO_BUCKET_NAME:   str = ""

    # -----< Directories >-----
    BASE_DIR:              Path = Path(__file__).parent.parent.parent
    DATA_DIR:              Path = BASE_DIR / "data"
    DOCUMENTS_DIR:         Path = DATA_DIR / "samples"
    VECTOR_STORE_DIR:      Path = DATA_DIR / "vector_store"  
    ROUTER_EMBEDDINGS_DIR: Path = VECTOR_STORE_DIR / "router_embeddings"
    BM25_CACHE_DIR:        Path = VECTOR_STORE_DIR / "bm25_store"
    FAISS_CACHE_DIR:       Path = VECTOR_STORE_DIR / "faiss_store"

    class Config:
        env_file          = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra             = "allow"
        case_sensitive    = True
    
    def model_post_init(self, __context):
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True) 
        self.ROUTER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)  


settings = Settings()