import time
import hashlib
import json
from typing import Dict, Any, Tuple, Optional

class RateLimiter:
    """Token-bucket rate limiter per API key."""
    def __init__(self, default_rpm: int = 60):
        self.default_rpm = default_rpm
        # Map: api_key -> (tokens, last_update_timestamp)
        self.buckets: Dict[str, Tuple[float, float]] = {}

    def is_allowed(self, api_key: str, rpm_limit: Optional[int] = None) -> Tuple[bool, int]:
        limit = rpm_limit or self.default_rpm
        now = time.time()
        tokens, last_updated = self.buckets.get(api_key, (float(limit), now))

        # Refill tokens according to elapsed time
        elapsed = now - last_updated
        refilled = tokens + elapsed * (limit / 60.0)
        current_tokens = min(float(limit), refilled)

        if current_tokens >= 1.0:
            self.buckets[api_key] = (current_tokens - 1.0, now)
            return True, int(current_tokens - 1.0)
        else:
            self.buckets[api_key] = (current_tokens, now)
            return False, 0

class ResponseCache:
    """Exact-match response cache with TTL."""
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        # key_hash -> (response_data, expiry_timestamp)
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    def _hash_request(self, model: str, messages: list, temperature: float = 0.7) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def get(self, model: str, messages: list, temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        key = self._hash_request(model, messages, temperature)
        if key in self._cache:
            data, expiry = self._cache[key]
            if time.time() < expiry:
                return data
            del self._cache[key]
        return None

    def set(self, model: str, messages: list, temperature: float, response: Dict[str, Any]):
        key = self._hash_request(model, messages, temperature)
        self._cache[key] = (response, time.time() + self.ttl)

    def clear(self):
        self._cache.clear()

def calculate_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates approximate USD cost based on token counts."""
    # Approximate pricing per 1M tokens ($/1M)
    pricing = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        "mock": (0.0, 0.0)
    }
    input_rate, output_rate = pricing.get(model, (0.50, 1.50))
    cost = (prompt_tokens * (input_rate / 1_000_000)) + (completion_tokens * (output_rate / 1_000_000))
    return round(cost, 6)

