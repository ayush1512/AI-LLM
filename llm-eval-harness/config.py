import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class EvalConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_under_test: str = os.getenv("MODEL_UNDER_TEST", "gpt-4o-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    pass_threshold: float = float(os.getenv("PASS_THRESHOLD", "0.70"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "5"))
    embedding_model: str = os.getenv("SEMANTIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

config = EvalConfig()

