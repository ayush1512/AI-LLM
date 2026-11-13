import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class SQLConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    database_path: str = os.getenv("DATABASE_PATH", "ecommerce.db")
    default_row_limit: int = int(os.getenv("DEFAULT_ROW_LIMIT", "20"))
    max_attempts: int = int(os.getenv("MAX_SELF_CORRECTION_ATTEMPTS", "3"))

config = SQLConfig()

