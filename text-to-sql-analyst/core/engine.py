import sqlite3
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from core.schema import get_scoped_schema, SCHEMA_METADATA
from core.validator import validate_and_sanitize_sql, clean_sql
from config import config

class TextToSQLEngine:
    """
    Enterprise Text-to-SQL Engine with:
    - Dynamic schema retrieval
    - Few-shot golden example injection
    - AST safety validation
    - Self-correction repair loop on database runtime errors
    - Read-only SQLite execution sandbox
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database_path
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "few_shot_queries.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _get_read_only_connection(self) -> sqlite3.Connection:
        # Open in URI read-only mode
        abs_path = os.path.abspath(self.db_path)
        uri = f"file:{abs_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def generate_sql(
        self,
        question: str,
        schema_context: str,
        error_context: Optional[str] = None,
        custom_api_key: Optional[str] = None
    ) -> str:
        api_key = custom_api_key or config.openai_api_key
        
        few_shots_text = "\n\n".join([
            f"Question: {ex['question']}\nSQL: {ex['sql']}"
            for ex in self.few_shot_examples[:3]
        ])

        system_prompt = (
            "You are an expert SQLite Data Analyst. Write pure SQLite queries based on the database schema.\n"
            "Rules:\n"
            "- Only generate a single valid SQLite SELECT query.\n"
            "- Do not wrap in markdown quotes if possible, or use ```sql.\n"
            "- Never generate UPDATE, INSERT, DROP, or DELETE queries.\n"
            "- Always use appropriate JOINs and column aliases."
        )

        repair_prompt = ""
        if error_context:
            repair_prompt = f"\n\nCRITICAL FIX: Your previous query produced this error:\n{error_context}\nPlease correct the column or table name and regenerate the SQL."

        user_prompt = (
            f"Database Schema:\n{schema_context}\n\n"
            f"Few-Shot Examples:\n{few_shots_text}\n\n"
            f"User Question: {question}{repair_prompt}\n\nSQL:"
        )

        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=config.openai_base_url)
                resp = client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return clean_sql(resp.choices[0].message.content)
            except Exception as e:
                pass

        # Offline heuristic fallback
        lowered = question.lower()
        if "customer" in lowered and ("top" in lowered or "spend" in lowered or "order" in lowered):
            return "SELECT c.customer_id, c.full_name, c.city, SUM(o.total_amount) AS total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'completed' GROUP BY c.customer_id, c.full_name ORDER BY total_spent DESC LIMIT 10;"
        elif "revenue" in lowered or "category" in lowered:
            return "SELECT cat.name AS category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS category_revenue FROM categories cat JOIN products p ON cat.category_id = p.category_id JOIN order_items oi ON p.product_id = oi.product_id GROUP BY cat.name ORDER BY category_revenue DESC LIMIT 10;"
        elif "stock" in lowered or "product" in lowered:
            return "SELECT product_id, name, price, stock FROM products WHERE stock < 50 ORDER BY stock ASC LIMIT 10;"
        else:
            return "SELECT order_id, customer_id, order_date, status, total_amount FROM orders ORDER BY order_date DESC LIMIT 10;"

    def execute_query(self, sql_query: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Executes SQL query in read-only sandbox mode and returns headers and rows."""
        with self._get_read_only_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [dict(row) for row in rows]
            return headers, data

    def run_analyst_pipeline(
        self,
        question: str,
        custom_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        schema_context = get_scoped_schema(question)
        
        attempt_history: List[Dict[str, Any]] = []
        current_error = None
        final_sql = ""
        success = False
        headers: List[str] = []
        data: List[Dict[str, Any]] = []

        for attempt in range(1, config.max_attempts + 1):
            attempt_start = time.time()
            # 1. Generate SQL
            raw_sql = self.generate_sql(
                question=question,
                schema_context=schema_context,
                error_context=current_error,
                custom_api_key=custom_api_key
            )

            # 2. Safety & AST Validation
            is_valid, sanitized_sql, val_error = validate_and_sanitize_sql(
                raw_sql, default_limit=config.default_row_limit
            )
            final_sql = sanitized_sql

            if not is_valid:
                current_error = f"Validation Error: {val_error}"
                attempt_history.append({
                    "attempt": attempt,
                    "sql": raw_sql,
                    "status": "validation_failed",
                    "error": val_error,
                    "duration_ms": round((time.time() - attempt_start) * 1000, 2)
                })
                continue

            # 3. Read-Only Execution
            try:
                headers, data = self.execute_query(sanitized_sql)
                success = True
                attempt_history.append({
                    "attempt": attempt,
                    "sql": sanitized_sql,
                    "status": "success",
                    "rows_returned": len(data),
                    "duration_ms": round((time.time() - attempt_start) * 1000, 2)
                })
                break
            except sqlite3.OperationalError as oe:
                current_error = f"SQLite OperationalError: {str(oe)}"
                attempt_history.append({
                    "attempt": attempt,
                    "sql": sanitized_sql,
                    "status": "execution_failed",
                    "error": str(oe),
                    "duration_ms": round((time.time() - attempt_start) * 1000, 2)
                })
            except Exception as ex:
                current_error = f"Database Error: {str(ex)}"
                attempt_history.append({
                    "attempt": attempt,
                    "sql": sanitized_sql,
                    "status": "execution_failed",
                    "error": str(ex),
                    "duration_ms": round((time.time() - attempt_start) * 1000, 2)
                })

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "question": question,
            "success": success,
            "final_sql": final_sql,
            "headers": headers,
            "data": data,
            "row_count": len(data),
            "attempts": attempt_history,
            "total_attempts": len(attempt_history),
            "total_latency_ms": total_time_ms,
            "schema_used": schema_context
        }

