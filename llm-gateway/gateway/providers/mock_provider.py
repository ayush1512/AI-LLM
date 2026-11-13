import time
from typing import List, Dict, Any
from gateway.providers.base import LLMProvider

class MockProvider(LLMProvider):
    """Deterministic, high-speed mock provider for testing and offline development."""
    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        last_msg = messages[-1]["content"] if messages else "Hello"
        mock_reply = f"[Mock Gateway Response] Echoing '{last_msg[:60]}...' via model={model}."
        
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        completion_tokens = len(mock_reply.split())

        return {
            "id": f"chatcmpl-mock-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "mock-model",
            "provider": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": mock_reply
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
