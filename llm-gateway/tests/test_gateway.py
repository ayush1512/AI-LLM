import unittest
import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.middleware import RateLimiter, ResponseCache, calculate_token_cost
from gateway.db import GatewayDB
from gateway.providers.mock_provider import MockProvider
from gateway.router import GatewayRouter

class TestLLMGateway(unittest.TestCase):
    def setUp(self):
        self.test_db_path = "test_gateway.db"
        self.db = GatewayDB(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_rate_limiter(self):
        limiter = RateLimiter(default_rpm=2)
        allowed1, _ = limiter.is_allowed("key1", rpm_limit=2)
        allowed2, _ = limiter.is_allowed("key1", rpm_limit=2)
        allowed3, _ = limiter.is_allowed("key1", rpm_limit=2)
        self.assertTrue(allowed1)
        self.assertTrue(allowed2)
        self.assertFalse(allowed3)

    def test_response_cache(self):
        cache = ResponseCache(ttl_seconds=10)
        messages = [{"role": "user", "content": "What is Python?"}]
        mock_resp = {"content": "Python is a programming language."}
        
        self.assertIsNone(cache.get("gpt-4o", messages, 0.7))
        cache.set("gpt-4o", messages, 0.7, mock_resp)
        cached = cache.get("gpt-4o", messages, 0.7)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["content"], "Python is a programming language.")

    def test_token_cost_calculation(self):
        cost = calculate_token_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        self.assertGreater(cost, 0.0)

    def test_mock_provider(self):
        provider = MockProvider()
        self.assertTrue(provider.is_available())
        resp = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-v1"
        )
        self.assertIn("choices", resp)
        self.assertEqual(resp["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", resp)

    def test_router_end_to_end(self):
        router = GatewayRouter(db=self.db)
        messages = [{"role": "user", "content": "Test prompt"}]
        # First call (uncached)
        res1 = router.route_completion(
            messages=messages,
            model="mock-model",
            api_key="gw-dev-test-key-123",
            preferred_provider="mock"
        )
        self.assertFalse(res1.get("cached", False))

        # Second call with same inputs should hit cache
        res2 = router.route_completion(
            messages=messages,
            model="mock-model",
            api_key="gw-dev-test-key-123",
            preferred_provider="mock"
        )
        self.assertTrue(res2.get("cached", False))

        # Check logs
        logs = self.db.get_recent_logs(limit=5)
        self.assertGreaterEqual(len(logs), 2)

if __name__ == "__main__":
    unittest.main()
