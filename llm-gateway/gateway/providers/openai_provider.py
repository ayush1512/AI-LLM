import time
import json
import urllib.request
from typing import List, Dict, Any
from gateway.providers.base import LLMProvider

class OpenAIProvider(LLMProvider):
    """Direct proxy to OpenAI and OpenAI-compatible endpoints."""
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60.0) as response:
            data = json.loads(response.read().decode('utf-8'))

        data["provider"] = self.name
        return data
