import re
import json
from typing import Optional, Dict, Any, List
from core.metrics.base import BaseMetric, MetricResult

class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Exact Match"

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        if not expected:
            return MetricResult(self.name, 1.0, True, "No reference expected output provided.")
        
        pred_norm = re.sub(r'\s+', ' ', prediction.strip().lower())
        exp_norm = re.sub(r'\s+', ' ', expected.strip().lower())
        
        passed = (pred_norm == exp_norm)
        score = 1.0 if passed else 0.0
        reason = "Output perfectly matches expected reference." if passed else "Output differs from reference string."
        return MetricResult(self.name, score, passed, reason)

class KeywordPresenceMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Keyword Presence"

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        meta = metadata or {}
        req_keywords: List[str] = meta.get("required_keywords", [])
        forbidden_keywords: List[str] = meta.get("forbidden_keywords", [])

        if not req_keywords and not forbidden_keywords:
            return MetricResult(self.name, 1.0, True, "No keyword constraints specified.")

        lowered = prediction.lower()
        
        # Check forbidden
        for f in forbidden_keywords:
            if f.lower() in lowered:
                return MetricResult(
                    self.name, 0.0, False,
                    f"Forbidden keyword '{f}' was detected in response.",
                    details={"violation": f}
                )

        # Check required
        if req_keywords:
            matched = [k for k in req_keywords if k.lower() in lowered]
            ratio = len(matched) / len(req_keywords)
            passed = (ratio >= 0.99)
            reason = f"Matched {len(matched)}/{len(req_keywords)} required keywords."
            return MetricResult(self.name, round(ratio, 2), passed, reason, details={"matched": matched, "missing": list(set(req_keywords) - set(matched))})

        return MetricResult(self.name, 1.0, True, "All keyword constraints met.")

class JsonSchemaMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "JSON Format & Keys"

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        meta = metadata or {}
        expected_keys: List[str] = meta.get("expected_keys", [])

        # Extract JSON block if wrapped in markdown
        cleaned = prediction.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return MetricResult(self.name, 0.0, False, f"Failed to parse valid JSON: {str(e)}")

        if not isinstance(parsed, dict):
            return MetricResult(self.name, 0.5, False, "Parsed JSON is not a key-value dictionary object.")

        if expected_keys:
            present_keys = [k for k in expected_keys if k in parsed]
            ratio = len(present_keys) / len(expected_keys)
            passed = (ratio >= 0.99)
            reason = f"Present keys: {present_keys} / {expected_keys}"
            return MetricResult(self.name, round(ratio, 2), passed, reason, details={"present": present_keys})

        return MetricResult(self.name, 1.0, True, "Valid JSON structure verified.")

class LengthBoundMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Length Constraint"

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        meta = metadata or {}
        max_words = meta.get("max_words")
        min_words = meta.get("min_words", 1)

        words = prediction.strip().split()
        word_count = len(words)

        if max_words and word_count > max_words:
            return MetricResult(
                self.name, 0.0, False,
                f"Response length ({word_count} words) exceeds maximum limit of {max_words} words.",
                details={"word_count": word_count, "max_words": max_words}
            )

        if word_count < min_words:
            return MetricResult(
                self.name, 0.0, False,
                f"Response length ({word_count} words) is below minimum of {min_words} words.",
                details={"word_count": word_count, "min_words": min_words}
            )

        return MetricResult(self.name, 1.0, True, f"Length constraint satisfied ({word_count} words).")

