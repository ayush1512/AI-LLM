import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    qdrant_location: str = os.getenv("QDRANT_LOCATION", ":memory:")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

settings = Settings()

