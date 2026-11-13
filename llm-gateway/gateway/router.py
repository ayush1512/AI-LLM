import time
from typing import Dict, Any, List, Optional
from gateway.providers.base import LLMProvider
from gateway.providers.mock_provider import MockProvider
from gateway.providers.openai_provider import OpenAIProvider
from gateway.providers.anthropic_provider import AnthropicProvider
from gateway.providers.gemini_provider import GeminiProvider
from gateway.providers.ollama_provider import OllamaProvider
from gateway.middleware import RateLimiter, ResponseCache, calculate_token_cost
from gateway.db import GatewayDB
from config import config

class GatewayRouter:
    """
    Central router orchestrating LLM request routing, rate limiting,
    caching, automatic fallback, and usage logging.
    """
    def __init__(self, db: Optional[GatewayDB] = None):
        self.db = db or GatewayDB(config.db_path)
        self.rate_limiter = RateLimiter(default_rpm=config.rate_limit_rpm)
        self.cache = ResponseCache(ttl_seconds=config.cache_ttl_seconds)
        
        # Initialize providers
        self.providers: Dict[str, LLMProvider] = {
            "mock": MockProvider(),
            "openai": OpenAIProvider(api_key=config.openai_api_key, base_url=config.openai_base_url),
            "anthropic": AnthropicProvider(api_key=config.anthropic_api_key, base_url=config.anthropic_base_url),
            "gemini": GeminiProvider(api_key=config.gemini_api_key, base_url=config.gemini_base_url),
            "ollama": OllamaProvider(base_url=config.ollama_base_url)
        }

    def route_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        api_key: str,
        preferred_provider: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Validate API Key
        key_data = self.db.validate_api_key(api_key)
        if not key_data and api_key != config.master_key:
            raise PermissionError("Invalid or inactive virtual API key.")

        # 2. Rate Limiting
        allowed, remaining = self.rate_limiter.is_allowed(api_key, rpm_limit=key_data.get("rate_limit_rpm") if key_data else None)
        if not allowed:
            raise ValueError("Rate limit exceeded for API key.")

        # 3. Response Caching
        if config.cache_enabled:
            cached_resp = self.cache.get(model=model, messages=messages, temperature=temperature)
            if cached_resp:
                latency = (time.time() - start_time) * 1000
                self.db.log_request(
                    api_key=api_key,
                    provider=cached_resp.get("provider", "cache"),
                    model=model,
                    prompt_tokens=cached_resp.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=cached_resp.get("usage", {}).get("completion_tokens", 0),
                    latency_ms=latency,
                    status="success_cached",
                    cost_usd=0.0,
                    cached=True
                )
                res_copy = dict(cached_resp)
                res_copy["cached"] = True
                return res_copy

        # 4. Determine Provider Candidate Order
        candidate_names = []
        if preferred_provider and preferred_provider in self.providers:
            candidate_names.append(preferred_provider)
        elif config.default_provider in self.providers:
            candidate_names.append(config.default_provider)
        
        # Append available fallbacks
        for name in ["mock", "openai", "anthropic", "gemini", "ollama"]:
            if name not in candidate_names:
                candidate_names.append(name)

        last_error = None
        for prov_name in candidate_names:
            provider = self.providers[prov_name]
            if not provider.is_available():
                continue

            try:
                response = provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    **kwargs
                )
                latency = (time.time() - start_time) * 1000
                
                usage = response.get("usage", {})
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)
                cost = calculate_token_cost(model, p_tok, c_tok)

                # Store in cache
                if config.cache_enabled:
                    self.cache.set(model=model, messages=messages, temperature=temperature, response=response)

                # Log usage
                self.db.log_request(
                    api_key=api_key,
                    provider=prov_name,
                    model=model,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=latency,
                    status="success",
                    cost_usd=cost,
                    cached=False
                )
                res_copy = dict(response)
                res_copy["cached"] = False
                return res_copy
            except Exception as e:
                last_error = e
                # Fallback to next provider
                continue

        # If all candidates failed
        latency = (time.time() - start_time) * 1000
        self.db.log_request(
            api_key=api_key,
            provider="none",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency,
            status="error",
            cost_usd=0.0,
            cached=False,
            error_message=str(last_error)
        )
        raise RuntimeError(f"All upstream LLM providers failed. Last error: {str(last_error)}")
