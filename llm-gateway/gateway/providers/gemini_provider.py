import time
import json
import urllib.request
from typing import List, Dict, Any
from gateway.providers.base import LLMProvider

class GeminiProvider(LLMProvider):
    """Adapter for Google Gemini REST API."""
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        target_model = model or "gemini-1.5-flash"
        url = f"{self.base_url}/models/{target_model}:generateContent?key={self.api_key}"
        
        contents = []
        for m in messages:
            role = "user" if m["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
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

        text_out = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text_out = parts[0].get("text", "")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return {
            "id": f"chatcmpl-gemini-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": target_model,
            "provider": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_out
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
