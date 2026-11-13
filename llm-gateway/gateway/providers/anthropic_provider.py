import time
import json
import urllib.request
from typing import List, Dict, Any
from gateway.providers.base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Adapter for Anthropic Claude Messages API."""
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        system_content = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content += m["content"] + "\n"
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model or "claude-3-5-sonnet-20241022",
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": temperature
        }
        if system_content.strip():
            payload["system"] = system_content.strip()

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60.0) as response:
            data = json.loads(response.read().decode('utf-8'))

        content = ""
        if "content" in data and len(data["content"]) > 0:
            content = data["content"][0].get("text", "")

        prompt_tokens = data.get("usage", {}).get("input_tokens", 0)
        completion_tokens = data.get("usage", {}).get("output_tokens", 0)

        return {
            "id": f"chatcmpl-anthropic-{data.get('id', '')}",
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
                    "finish_reason": data.get("stop_reason", "stop")
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
