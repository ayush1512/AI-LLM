import re
from typing import List, Optional
from core.guardrails.base import BaseRail, RailResult

class PIIDetector(BaseRail):
    """Detects and redacts sensitive Personally Identifiable Information (PII)."""
    @property
    def name(self) -> str:
        return "Input: PII Detection"

    # Regex patterns for high-risk PII
    PATTERNS = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[ -]?){3}\d{4}\b',
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
        "PHONE": r'\b(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b'
    }

    def check(self, text: str, redact: bool = True) -> RailResult:
        detected = []
        sanitized = text
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, sanitized)
            if matches:
                detected.append(pii_type)
                if redact:
                    sanitized = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized)

        if detected:
            if "SSN" in detected or "CREDIT_CARD" in detected:
                # Critical PII triggers a block
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason=f"High-risk PII detected: {', '.join(detected)}.",
                    sanitized_text=sanitized,
                    metadata={"detected": detected}
                )
            else:
                # Moderate PII is sanitized/redacted
                return RailResult(
                    passed=True,
                    rail_name=self.name,
                    action="redact",
                    reason=f"PII masked: {', '.join(detected)}.",
                    sanitized_text=sanitized,
                    metadata={"detected": detected}
                )

        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

class PromptInjectionDetector(BaseRail):
    """Detects jailbreaks, delimiter manipulation, and prompt injection attempts."""
    @property
    def name(self) -> str:
        return "Input: Prompt Injection"

    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)',
        r'you\s+are\s+now\s+(DAN|unrestricted|in\s+developer\s+mode)',
        r'system\s+prompt\s+(reveal|leak|print|show)',
        r'disregard\s+(all\s+)?guardrails',
        r'override\s+safety\s+protocols',
        r'pretend\s+you\s+have\s+no\s+(rules|restrictions)',
        r'bypass\s+(content\s+filter|safety|filter)'
    ]

    def check(self, text: str) -> RailResult:
        lowered = text.lower()
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, lowered):
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason="Potential adversarial prompt injection detected.",
                    sanitized_text=text,
                    metadata={"matched_pattern": pat}
                )
        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

class TopicGuard(BaseRail):
    """Ensures conversation remains strictly within authorized organizational scope."""
    @property
    def name(self) -> str:
        return "Input: Topic Boundary"

    DISALLOWED_PATTERNS = [
        r'\b(election|political\s+campaign|vote\s+for|politician|congress|president\s+debate)\b',
        r'\b(medical\s+diagnosis|prescribe\s+medication|clinical\s+dosage|cure\s+cancer)\b',
        r'\b(how\s+to\s+hack|create\s+malware|synthesize\s+explosive|ddos\s+attack)\b'
    ]

    def __init__(self, allowed_topics: Optional[List[str]] = None):
        self.allowed_topics = allowed_topics or ["finance", "policy", "operations", "calculations"]

    def check(self, text: str) -> RailResult:
        lowered = text.lower()
        for pat in self.DISALLOWED_PATTERNS:
            if re.search(pat, lowered):
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason="Input falls outside the approved corporate scope (e.g. politics, medical advice, cyber warfare).",
                    sanitized_text=text
                )
        return RailResult(passed=True, rail_name=self.name, action="allow", sanitized_text=text)

