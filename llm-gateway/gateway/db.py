import sqlite3
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class GatewayDB:
    """Manages SQLite persistence for gateway usage logs, API keys, and cache."""
    def __init__(self, db_path: str = "gateway.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Usage logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                status TEXT NOT NULL,
                cost_usd REAL DEFAULT 0.0,
                cached INTEGER DEFAULT 0,
                error_message TEXT
            )
            """)
            
            # Virtual API Keys table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS virtual_keys (
                key TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                rate_limit_rpm INTEGER DEFAULT 60,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
            """)

            # Seed default demo key if none exist
            cursor.execute("SELECT COUNT(*) FROM virtual_keys")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO virtual_keys (key, project_name, rate_limit_rpm, created_at, is_active)
                VALUES ('gw-dev-test-key-123', 'Production Assistant', 100, datetime('now'), 1)
                """)
            conn.commit()

    def log_request(
        self,
        api_key: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str,
        cost_usd: float = 0.0,
        cached: bool = False,
        error_message: Optional[str] = None
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO usage_logs (
                timestamp, api_key, provider, model, prompt_tokens,
                completion_tokens, total_tokens, latency_ms, status,
                cost_usd, cached, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                api_key,
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                latency_ms,
                status,
                cost_usd,
                1 if cached else 0,
                error_message
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM usage_logs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_aggregated_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as total_cost,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
                COALESCE(SUM(cached), 0) as cache_hits
            FROM usage_logs
            """)
            row = cursor.fetchone()
            return dict(row) if row else {
                "total_requests": 0, "total_tokens": 0,
                "total_cost": 0.0, "avg_latency_ms": 0.0, "cache_hits": 0
            }

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM virtual_keys WHERE key = ? AND is_active = 1", (api_key,))
            row = cursor.fetchone()
            return dict(row) if row else None
