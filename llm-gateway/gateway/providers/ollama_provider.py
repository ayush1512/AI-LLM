import time
import json
import urllib.request
from typing import List, Dict, Any
from gateway.providers.base import LLMProvider

class OllamaProvider(LLMProvider):
    """Adapter for local Ollama instances."""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        # Check if URL is specified; doesn't require cloud API key
        return bool(self.base_url)

    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model or "llama3",
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60.0) as response:
            data = json.loads(response.read().decode('utf-8'))

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return {
            "id": f"chatcmpl-ollama-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "provider": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
