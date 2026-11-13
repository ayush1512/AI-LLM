import re
from typing import Tuple, Optional

BLOCKED_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "ALTER", "INSERT",
    "TRUNCATE", "CREATE", "EXEC", "EXECUTE", "REPLACE",
    "PRAGMA", "ATTACH", "DETACH", "GRANT", "REVOKE"
}

def clean_sql(raw_sql: str) -> str:
    """Removes markdown code fences and extraneous text from LLM response."""
    cleaned = raw_sql.strip()
    if cleaned.startswith("```sql"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def validate_and_sanitize_sql(sql_query: str, default_limit: int = 20) -> Tuple[bool, str, Optional[str]]:
    """
    Validates that SQL is strictly a read-only SELECT query,
    has no destructive statements, and enforces a safety LIMIT clause.
    Returns: (is_valid, sanitized_sql, error_message)
    """
    cleaned = clean_sql(sql_query)
    if not cleaned:
        return False, "", "Empty SQL query generated."

    # Remove single line and block comments to prevent bypasses
    uncommented = re.sub(r'--.*?\n', ' ', cleaned)
    uncommented = re.sub(r'/\*.*?\*/', ' ', uncommented, flags=re.DOTALL)

    # Tokenize words
    tokens = [t.upper() for t in re.findall(r'\b[A-Za-z_]+\b', uncommented)]
    
    # Check for blocked keywords
    for t in tokens:
        if t in BLOCKED_KEYWORDS:
            return False, cleaned, f"Security Violation: '{t}' operations are forbidden. Only read-only SELECT queries are permitted."

    # Must start with SELECT or WITH (for CTEs)
    if not (tokens and tokens[0] in ["SELECT", "WITH"]):
        return False, cleaned, "Invalid SQL: Query must begin with a SELECT or WITH statement."

    # Enforce LIMIT clause
    if "LIMIT" not in tokens:
        # Strip trailing semicolon if present
        query_body = cleaned.rstrip(";").strip()
        cleaned = f"{query_body} LIMIT {default_limit};"

    return True, cleaned, None

