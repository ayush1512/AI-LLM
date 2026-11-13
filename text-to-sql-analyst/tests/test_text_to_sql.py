import unittest
import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.validator import validate_and_sanitize_sql, clean_sql
from core.schema import get_scoped_schema
from core.engine import TextToSQLEngine
from data.seed_db import create_and_seed_db

class TestTextToSQL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = "test_ecommerce.db"
        create_and_seed_db(cls.test_db)
        cls.engine = TextToSQLEngine(db_path=cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_sql_validator_safe_query(self):
        sql = "SELECT full_name, email FROM customers"
        is_valid, sanitized, err = validate_and_sanitize_sql(sql, default_limit=15)
        self.assertTrue(is_valid)
        self.assertIn("LIMIT 15;", sanitized)
        self.assertIsNone(err)

    def test_sql_validator_blocks_drop(self):
        sql = "DROP TABLE customers;"
        is_valid, _, err = validate_and_sanitize_sql(sql)
        self.assertFalse(is_valid)
        self.assertIn("DROP", err)

    def test_sql_validator_blocks_delete(self):
        sql = "DELETE FROM orders WHERE order_id = 1;"
        is_valid, _, err = validate_and_sanitize_sql(sql)
        self.assertFalse(is_valid)
        self.assertIn("DELETE", err)

    def test_sql_cleaner_markdown(self):
        raw = "```sql\nSELECT * FROM products;\n```"
        cleaned = clean_sql(raw)
        self.assertEqual(cleaned, "SELECT * FROM products;")

    def test_schema_scoping(self):
        context = get_scoped_schema("What are the product categories and inventory?")
        self.assertIn("categories", context)
        self.assertIn("products", context)

    def test_engine_execution(self):
        sql = "SELECT category_id, name FROM categories ORDER BY category_id ASC LIMIT 5;"
        headers, rows = self.engine.execute_query(sql)
        self.assertEqual(headers, ["category_id", "name"])
        self.assertGreaterEqual(len(rows), 4)
        self.assertEqual(rows[0]["name"], "Electronics")

    def test_pipeline_end_to_end(self):
        res = self.engine.run_analyst_pipeline("Show top customers by order spend")
        self.assertTrue(res["success"])
        self.assertGreater(res["row_count"], 0)
        self.assertIn("final_sql", res)

if __name__ == "__main__":
    unittest.main()

