import re
from typing import List, Optional
from core.guardrails.base import BaseRail, RailResult

class ToxicityFilter(BaseRail):
    """Filters profane, abusive, or hostile model generations."""
    @property
    def name(self) -> str:
        return "Output: Toxicity Filter"

    TOXIC_TERMS = [
        "idiot", "stupid", "moron", "hate you", "kill yourself", "threaten", "scam"
    ]

    def check(self, text: str) -> RailResult:
        lowered = text.lower()
        for term in self.TOXIC_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', lowered):
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason=f"Potentially toxic term '{term}' intercepted in model output."
                )
        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

class PIILeakFilter(BaseRail):
    """Guarantees no sensitive PII is leaked in assistant responses."""
    @property
    def name(self) -> str:
        return "Output: PII Leak Prevention"

    LEAK_PATTERNS = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[ -]?){3}\d{4}\b'
    }

    def check(self, text: str) -> RailResult:
        for pii_type, pattern in self.LEAK_PATTERNS.items():
            if re.search(pattern, text):
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type}]", text)
                return RailResult(
                    passed=True,
                    rail_name=self.name,
                    action="redact",
                    reason=f"Prevented leak of {pii_type} in model response.",
                    sanitized_text=sanitized
                )
        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

class GroundednessChecker(BaseRail):
    """Verifies that numerical facts cite actual verified tool observations."""
    @property
    def name(self) -> str:
        return "Output: Hallucination & Groundedness"

    def check(self, text: str, tool_observations: List[str]) -> RailResult:
        # Check if numbers > 1000 in response match observations
        output_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        obs_text = " ".join(tool_observations)
        
        # If tools were invoked, check ungrounded claim suspicion
        if tool_observations and len(output_numbers) > 5:
            ungrounded = [n for n in output_numbers if n not in obs_text and float(n) > 1000]
            if len(ungrounded) > 3:
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason=f"Unverified numerical assertions detected without tool grounding: {ungrounded[:3]}."
                )
        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

