import math
from typing import Dict, Any

POLICY_KNOWLEDGE_BASE = {
    "travel": "Employees may expense domestic flights in economy class. Daily meals cap is $75/day. Hotel stays up to $220/night are pre-approved.",
    "equipment": "Hardware refresh cycle is 36 months. Remote employees receive a $1,000 home office setup stipend upon hiring.",
    "leave": "Full-time employees receive 20 days paid annual leave and 10 days sick leave.",
    "wfh": "Hybrid policy: minimum 2 days in office per week, 3 days remote permitted with manager alignment."
}

CURRENCY_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.78,
    "JPY": 154.2,
    "INR": 83.5
}

def execute_calculator(expression: str) -> str:
    """Safe arithmetic evaluator for expressions."""
    # Whitelisted math environment
    safe_env = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow
    }
    try:
        # Note: ArgumentValidationRail already checked for dangerous tokens
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return str(round(float(result), 4))
    except Exception as e:
        return f"Calculation Error: {str(e)}"

def execute_currency_converter(amount: float, from_curr: str, to_curr: str) -> str:
    """Convert currency using base exchange rates."""
    from_c = from_curr.upper().strip()
    to_c = to_curr.upper().strip()
    if from_c not in CURRENCY_RATES or to_c not in CURRENCY_RATES:
        return f"Error: Unsupported currency code. Available: {list(CURRENCY_RATES.keys())}"
    
    usd_val = amount / CURRENCY_RATES[from_c]
    target_val = usd_val * CURRENCY_RATES[to_c]
    return f"{amount} {from_c} = {target_val:.2f} {to_c} (Exchange rate: 1 {from_c} = {CURRENCY_RATES[to_c]/CURRENCY_RATES[from_c]:.4f} {to_c})"

def execute_company_policy_lookup(topic: str) -> str:
    """Query company policy document store."""
    t_clean = topic.lower().strip()
    for key, text in POLICY_KNOWLEDGE_BASE.items():
        if key in t_clean or t_clean in key:
            return f"Company Policy ({key}): {text}"
    return f"No specific policy clause found for '{topic}'. Standard operations guidelines apply."

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic string e.g. '150 * 1.08 + 45'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "currency_converter",
            "description": "Converts currency between USD, EUR, GBP, JPY, and INR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Monetary amount to convert"},
                    "from_curr": {"type": "string", "description": "Source ISO currency code e.g. USD"},
                    "to_curr": {"type": "string", "description": "Target ISO currency code e.g. EUR"}
                },
                "required": ["amount", "from_curr", "to_curr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "company_policy_lookup",
            "description": "Look up enterprise policies on travel, equipment, leave, and remote work (wfh).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Policy domain e.g. travel, equipment, leave, wfh"}
                },
                "required": ["topic"]
            }
        }
    }
]

