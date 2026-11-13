from typing import List, Dict, Any, Tuple
from core.guardrails.base import RailResult
from core.guardrails.input_rails import PIIDetector, PromptInjectionDetector, TopicGuard
from core.guardrails.tool_rails import ToolAuthorizationRail, ArgumentValidationRail
from core.guardrails.output_rails import ToxicityFilter, PIILeakFilter, GroundednessChecker
from config import config

class GuardrailsPipeline:
    """Orchestrates multi-layered input, tool, and output rails."""
    def __init__(self):
        self.pii_rail = PIIDetector()
        self.injection_rail = PromptInjectionDetector()
        self.topic_rail = TopicGuard(allowed_topics=config.allowed_topics)
        
        self.tool_auth_rail = ToolAuthorizationRail(
            allowed_tools={"calculator", "currency_converter", "company_policy_lookup"}
        )
        self.arg_val_rail = ArgumentValidationRail()
        
        self.toxic_rail = ToxicityFilter()
        self.leak_rail = PIILeakFilter()
        self.grounded_rail = GroundednessChecker()

    def inspect_input(
        self,
        user_input: str,
        enabled_rails: Dict[str, bool]
    ) -> Tuple[bool, str, List[RailResult]]:
        results: List[RailResult] = []
        current_text = user_input

        # 1. Prompt Injection
        if enabled_rails.get("injection", True):
            r = self.injection_rail.check(current_text)
            results.append(r)
            if not r.passed:
                return False, f"🛡️ **Security Alert**: {r.reason}", results

        # 2. Topic Guard
        if enabled_rails.get("topic", True):
            r = self.topic_rail.check(current_text)
            results.append(r)
            if not r.passed:
                return False, f"🛡️ **Policy Alert**: {r.reason}", results

        # 3. PII Detection
        if enabled_rails.get("pii", True):
            r = self.pii_rail.check(current_text)
            results.append(r)
            if not r.passed:
                return False, f"🛡️ **Privacy Alert**: {r.reason}", results
            if r.sanitized_text:
                current_text = r.sanitized_text

        return True, current_text, results

    def inspect_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        enabled: bool = True
    ) -> Tuple[bool, str, List[RailResult]]:
        if not enabled:
            return True, "", []

        results = []
        # Auth check
        r_auth = self.tool_auth_rail.check(tool_name)
        results.append(r_auth)
        if not r_auth.passed:
            return False, r_auth.reason, results

        # Arg check
        r_arg = self.arg_val_rail.check(tool_name, tool_args)
        results.append(r_arg)
        if not r_arg.passed:
            return False, r_arg.reason, results

        return True, "", results

    def inspect_output(
        self,
        output_text: str,
        tool_observations: List[str],
        enabled_rails: Dict[str, bool]
    ) -> Tuple[bool, str, List[RailResult]]:
        results = []
        current_text = output_text

        # Toxicity
        if enabled_rails.get("toxicity", True):
            r = self.toxic_rail.check(current_text)
            results.append(r)
            if not r.passed:
                return False, "🛡️ Response intercepted: Output violated safety policies.", results

        # PII Leak
        if enabled_rails.get("pii_leak", True):
            r = self.leak_rail.check(current_text)
            results.append(r)
            if r.sanitized_text:
                current_text = r.sanitized_text

        # Groundedness
        if enabled_rails.get("groundedness", True):
            r = self.grounded_rail.check(current_text, tool_observations)
            results.append(r)
            if not r.passed:
                return False, f"🛡️ Response blocked: {r.reason}", results

        return True, current_text, results

