import json
import time
from typing import List, Dict, Any, Optional
from core.guardrails.pipeline import GuardrailsPipeline
from core.guardrails.base import RailResult
from core.tools import (
    execute_calculator,
    execute_currency_converter,
    execute_company_policy_lookup,
    TOOLS_SCHEMA
)
from config import config

class GuardedReActAgent:
    """
    Autonomous ReAct Agent wrapped with multi-tier Guardrails.
    Inspects input prompts, enforces tool execution authorization,
    and sanitizes final output.
    """
    def __init__(self, pipeline: Optional[GuardrailsPipeline] = None):
        self.pipeline = pipeline or GuardrailsPipeline()

    def run_turn(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        enabled_rails: Dict[str, bool],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        audit_trail: List[Dict[str, Any]] = []
        tool_observations: List[str] = []

        # -------------------------------------------------------------
        # 1. INPUT GUARDRAILS
        # -------------------------------------------------------------
        input_passed, sanitized_prompt, input_results = self.pipeline.inspect_input(
            user_message, enabled_rails
        )
        for r in input_results:
            audit_trail.append({
                "stage": "input",
                "rail": r.rail_name,
                "passed": r.passed,
                "action": r.action,
                "reason": r.reason
            })

        if not input_passed:
            return {
                "response": sanitized_prompt,
                "blocked": True,
                "stage_blocked": "input",
                "audit_trail": audit_trail,
                "tool_calls": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        # -------------------------------------------------------------
        # 2. AGENT DECISION / TOOL CALLING LOOP
        # -------------------------------------------------------------
        chosen_key = api_key or config.openai_api_key
        agent_reply = ""
        tool_call_records = []

        if chosen_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=chosen_key, base_url=config.openai_base_url)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional enterprise operations assistant. "
                            "You have access to tools: calculator, currency_converter, company_policy_lookup. "
                            "Use tools whenever computations or policy lookups are required. "
                            "Provide clear and concise answers."
                        )
                    }
                ]
                messages.extend(conversation_history)
                messages.append({"role": "user", "content": sanitized_prompt})

                # First call: check for tool calls
                resp = client.chat.completions.create(
                    model=config.model_name,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.2
                )
                msg = resp.choices[0].message

                if msg.tool_calls:
                    messages.append(msg)
                    for tc in msg.tool_calls:
                        t_name = tc.function.name
                        t_args = json.loads(tc.function.arguments)

                        # Tool Guardrail check
                        tool_ok, tool_reason, tool_rails = self.pipeline.inspect_tool_call(
                            t_name, t_args, enabled=enabled_rails.get("tool", True)
                        )
                        for r in tool_rails:
                            audit_trail.append({
                                "stage": "tool",
                                "rail": r.rail_name,
                                "passed": r.passed,
                                "action": r.action,
                                "reason": r.reason
                            })

                        if not tool_ok:
                            tool_obs = f"Guardrail Rejection: {tool_reason}"
                        else:
                            # Execute verified tool
                            if t_name == "calculator":
                                tool_obs = execute_calculator(t_args.get("expression", ""))
                            elif t_name == "currency_converter":
                                tool_obs = execute_currency_converter(
                                    t_args.get("amount", 0),
                                    t_args.get("from_curr", "USD"),
                                    t_args.get("to_curr", "EUR")
                                )
                            elif t_name == "company_policy_lookup":
                                tool_obs = execute_company_policy_lookup(t_args.get("topic", ""))
                            else:
                                tool_obs = "Error: Unknown tool."

                        tool_observations.append(tool_obs)
                        tool_call_records.append({"tool": t_name, "args": t_args, "output": tool_obs})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_obs
                        })

                    # Second call: LLM synthesizes final answer
                    final_resp = client.chat.completions.create(
                        model=config.model_name,
                        messages=messages,
                        temperature=0.2
                    )
                    agent_reply = final_resp.choices[0].message.content
                else:
                    agent_reply = msg.content or "I have processed your request."
            except Exception as e:
                agent_reply = f"System Error during LLM turn: {str(e)}"
        else:
            # Deterministic agent simulation mode (offline / without API key)
            lowered = sanitized_prompt.lower()
            if "convert" in lowered or "usd" in lowered or "eur" in lowered:
                # Trigger currency tool
                t_ok, _, t_rails = self.pipeline.inspect_tool_call("currency_converter", {"amount": 100, "from_curr": "USD", "to_curr": "EUR"})
                for r in t_rails:
                    audit_trail.append({"stage": "tool", "rail": r.rail_name, "passed": r.passed, "action": r.action, "reason": r.reason})
                obs = execute_currency_converter(100, "USD", "EUR")
                tool_observations.append(obs)
                tool_call_records.append({"tool": "currency_converter", "args": {"amount": 100, "from_curr": "USD", "to_curr": "EUR"}, "output": obs})
                agent_reply = f"Here is the conversion result: {obs}"
            elif "policy" in lowered or "travel" in lowered or "leave" in lowered or "wfh" in lowered:
                t_ok, _, t_rails = self.pipeline.inspect_tool_call("company_policy_lookup", {"topic": "travel"})
                for r in t_rails:
                    audit_trail.append({"stage": "tool", "rail": r.rail_name, "passed": r.passed, "action": r.action, "reason": r.reason})
                obs = execute_company_policy_lookup("travel")
                tool_observations.append(obs)
                tool_call_records.append({"tool": "company_policy_lookup", "args": {"topic": "travel"}, "output": obs})
                agent_reply = f"According to our corporate guidelines: {obs}"
            else:
                agent_reply = f"I am your enterprise assistant. Your input '{sanitized_prompt}' was processed safely through all guardrails."

        # -------------------------------------------------------------
        # 3. OUTPUT GUARDRAILS
        # -------------------------------------------------------------
        output_passed, final_text, output_results = self.pipeline.inspect_output(
            agent_reply, tool_observations, enabled_rails
        )
        for r in output_results:
            audit_trail.append({
                "stage": "output",
                "rail": r.rail_name,
                "passed": r.passed,
                "action": r.action,
                "reason": r.reason
            })

        if not output_passed:
            return {
                "response": final_text,
                "blocked": True,
                "stage_blocked": "output",
                "audit_trail": audit_trail,
                "tool_calls": tool_call_records,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        return {
            "response": final_text,
            "blocked": False,
            "stage_blocked": None,
            "audit_trail": audit_trail,
            "tool_calls": tool_call_records,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

