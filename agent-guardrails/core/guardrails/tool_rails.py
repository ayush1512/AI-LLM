import re
from typing import Dict, Any, Set
from core.guardrails.base import BaseRail, RailResult

class ToolAuthorizationRail(BaseRail):
    """Enforces strict whitelisting of callable tools."""
    @property
    def name(self) -> str:
        return "Tool: Authorization Whitelist"

    def __init__(self, allowed_tools: Set[str]):
        self.allowed_tools = allowed_tools

    def check(self, tool_name: str) -> RailResult:
        if tool_name not in self.allowed_tools:
            return RailResult(
                passed=False,
                rail_name=self.name,
                action="block",
                reason=f"Unauthorized tool invocation attempt: '{tool_name}'. Allowed: {list(self.allowed_tools)}"
            )
        return RailResult(passed=True, rail_name=self.name, action="allow")

class ArgumentValidationRail(BaseRail):
    """Validates and sanitizes parameters passed into tool execution."""
    @property
    def name(self) -> str:
        return "Tool: Argument Validation"

    DANGEROUS_EXPR = [
        "__", "import", "exec", "eval", "os.", "sys.", "subprocess", "open(", "globals", "locals"
    ]

    def check(self, tool_name: str, args: Dict[str, Any]) -> RailResult:
        if tool_name == "calculator":
            expr = str(args.get("expression", ""))
            for d in self.DANGEROUS_EXPR:
                if d in expr:
                    return RailResult(
                        passed=False,
                        rail_name=self.name,
                        action="block",
                        reason=f"Dangerous code pattern '{d}' detected in math expression."
                    )
            # Only allow arithmetic characters and numbers
            if not re.match(r'^[0-9+\-*/().\s^%]+$', expr):
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason="Expression contains invalid non-arithmetic characters."
                )

        elif tool_name == "currency_converter":
            amount = args.get("amount")
            if not isinstance(amount, (int, float)) or amount < 0:
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason="Currency amount must be a positive number."
                )

        elif tool_name == "company_policy_lookup":
            topic = str(args.get("topic", ""))
            if "../" in topic or "DROP TABLE" in topic.upper():
                return RailResult(
                    passed=False,
                    rail_name=self.name,
                    action="block",
                    reason="Path traversal or injection syntax detected in policy query."
                )

        return RailResult(passed=True, rail_name=self.name, action="allow")

