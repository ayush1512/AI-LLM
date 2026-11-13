import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class AgentConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    enable_pii_rail: bool = os.getenv("ENABLE_PII_RAIL", "true").lower() == "true"
    enable_prompt_injection_rail: bool = os.getenv("ENABLE_PROMPT_INJECTION_RAIL", "true").lower() == "true"
    enable_topic_rail: bool = os.getenv("ENABLE_TOPIC_RAIL", "true").lower() == "true"
    enable_tool_rail: bool = os.getenv("ENABLE_TOOL_RAIL", "true").lower() == "true"
    enable_output_rail: bool = os.getenv("ENABLE_OUTPUT_RAIL", "true").lower() == "true"

    allowed_topics: List[str] = field(default_factory=lambda: [
        "finance", "business operations", "expenses", "company policy", "budgeting", "analytics", "calculations"
    ])

config = AgentConfig()

