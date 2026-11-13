import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class GatewayConfig:
    host: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port: int = int(os.getenv("GATEWAY_PORT", "8000"))
    master_key: str = os.getenv("GATEWAY_MASTER_KEY", "gw-master-secret-key")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    default_provider: str = os.getenv("DEFAULT_PROVIDER", "mock")
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    db_path: str = os.getenv("DATABASE_PATH", "gateway.db")

config = GatewayConfig()

