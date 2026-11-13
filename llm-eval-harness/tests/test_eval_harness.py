import unittest
import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dataset import load_dataset_from_jsonl, EvalCase
from core.metrics.deterministic import (
    ExactMatchMetric, KeywordPresenceMetric, JsonSchemaMetric, LengthBoundMetric
)
from core.metrics.semantic import CosineSimilarityMetric
from core.metrics.llm_judge import LLMJudgeMetric, FAITHFULNESS_RUBRIC
from core.runner import EvaluationRunner

class TestEvalHarness(unittest.TestCase):
    def test_exact_match(self):
        m = ExactMatchMetric()
        res_pass = m.evaluate("Paris", expected="paris")
        self.assertTrue(res_pass.passed)
        self.assertEqual(res_pass.score, 1.0)
        
        res_fail = m.evaluate("London", expected="Paris")
        self.assertFalse(res_fail.passed)
        self.assertEqual(res_fail.score, 0.0)

    def test_keyword_presence(self):
        m = KeywordPresenceMetric()
        meta = {"required_keywords": ["python", "asyncio"], "forbidden_keywords": ["deprecated"]}
        
        # Pass
        res1 = m.evaluate("Python provides native asyncio support.", metadata=meta)
        self.assertTrue(res1.passed)
        
        # Forbidden detected
        res2 = m.evaluate("Python provides deprecated asyncio support.", metadata=meta)
        self.assertFalse(res2.passed)
        self.assertEqual(res2.score, 0.0)

    def test_json_schema_metric(self):
        m = JsonSchemaMetric()
        meta = {"expected_keys": ["name", "score"]}
        
        # Valid JSON with keys
        res1 = m.evaluate('{"name": "test", "score": 95}', metadata=meta)
        self.assertTrue(res1.passed)
        
        # Missing key
        res2 = m.evaluate('{"name": "test"}', metadata=meta)
        self.assertFalse(res2.passed)
        self.assertEqual(res2.score, 0.5)

    def test_length_bound_metric(self):
        m = LengthBoundMetric()
        meta = {"max_words": 5}
        res_pass = m.evaluate("Short concise sentence.", metadata=meta)
        self.assertTrue(res_pass.passed)

        res_fail = m.evaluate("This sentence has way too many unnecessary words in it.", metadata=meta)
        self.assertFalse(res_fail.passed)

    def test_semantic_cosine(self):
        m = CosineSimilarityMetric(threshold=0.6)
        res = m.evaluate("The capital of France is Paris.", expected="Paris is the French capital.")
        self.assertGreater(res.score, 0.5)

    def test_dataset_loader(self):
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite.jsonl")
        cases = load_dataset_from_jsonl(dataset_path)
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(cases[0].case_id, "eval-qa-01")

    def test_runner_batch_execution(self):
        cases = [
            EvalCase("case-1", "What port is PostgreSQL?", expected_output="5432", metadata={"required_keywords": ["5432"]})
        ]
        metrics = [KeywordPresenceMetric(), ExactMatchMetric()]
        runner = EvaluationRunner(metrics=metrics)
        report = runner.run_suite(cases)
        
        self.assertEqual(report["cases_evaluated"], 1)
        self.assertIn("scorecard", report)
        self.assertIn("Keyword Presence", report["scorecard"])

if __name__ == "__main__":
    unittest.main()

