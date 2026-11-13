import streamlit as st
import pandas as pd
from gateway.router import GatewayRouter
from gateway.db import GatewayDB
from config import config

st.set_page_config(
    page_title="LLM Gateway Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        padding: 16px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .badge-cache {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

if "router" not in st.session_state:
    st.session_state.router = GatewayRouter()
if "db" not in st.session_state:
    st.session_state.db = st.session_state.router.db

st.title("🛡️ Unified LLM API Gateway")
st.caption("Centralized multi-provider proxy (OpenAI, Anthropic, Gemini, Ollama) with rate limiting, caching, & fallback.")

# Top Metrics Row
stats = st.session_state.db.get_aggregated_stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Requests", stats.get("total_requests", 0))
c2.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
c3.metric("Estimated Cost", f"${stats.get('total_cost', 0.0):.4f}")
c4.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0.0):.1f} ms")
c5.metric("Cache Hits", stats.get("cache_hits", 0))

st.divider()

tab_playground, tab_logs, tab_keys, tab_arch = st.tabs(["⚡ Gateway Playground", "📋 Usage & Audit Logs", "🔑 Virtual API Keys", "🏗️ Architecture"])

with tab_playground:
    st.subheader("Test Request via Gateway Proxy")
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        v_key = st.text_input("Virtual API Key", value="gw-dev-test-key-123")
        pref_provider = st.selectbox("Target Provider", ["mock", "openai", "anthropic", "gemini", "ollama"])
        model_name = st.selectbox("Model", ["mock-model", "gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-flash", "llama3"])
        temp = st.slider("Temperature", 0.0, 1.0, 0.7)
        prompt_text = st.text_area("Prompt", "Explain the difference between bi-encoders and cross-encoders.")
        submit_btn = st.button("Send Request", type="primary", use_container_width=True)

    with col_p2:
        st.write("**Gateway Response & Telemetry**")
        if submit_btn:
            with st.spinner("Routing through gateway middleware..."):
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    # Execute synchronous call
                    res = st.session_state.router.route_completion(
                        messages=messages,
                        model=model_name,
                        api_key=v_key,
                        preferred_provider=pref_provider,
                        temperature=temp
                    )
                    
                    cached = res.get("cached", False)
                    cache_badge = "🟢 CACHE HIT" if cached else "🔵 LIVE UPSTREAM CALL"
                    st.markdown(f"**Status:** {cache_badge} | **Provider:** `{res.get('provider')}` | **Model:** `{res.get('model')}`")
                    
                    st.subheader("Content")
                    st.info(res["choices"][0]["message"]["content"])
                    
                    u = res.get("usage", {})
                    st.code(f"Usage: {u.get('prompt_tokens', 0)} prompt tokens + {u.get('completion_tokens', 0)} completion tokens = {u.get('total_tokens', 0)} total", language="text")
                except Exception as e:
                    st.error(f"Gateway Error: {str(e)}")

with tab_logs:
    st.subheader("Recent Request Logs")
    raw_logs = st.session_state.db.get_recent_logs(limit=25)
    if raw_logs:
        df = pd.DataFrame(raw_logs)
        st.dataframe(
            df[["id", "timestamp", "provider", "model", "total_tokens", "latency_ms", "cached", "status", "cost_usd"]],
            use_container_width=True
        )
    else:
        st.info("No logs recorded yet. Send a request in the playground!")

with tab_keys:
    st.subheader("Virtual Keys & Quotas")
    st.caption("Virtual API keys isolate teams/projects, enforce rate limits, and track spend.")
    with st.expander("➕ Register New Virtual Key"):
        with st.form("new_key_form"):
            new_key = st.text_input("New API Key", value="gw-client-finance-app")
            new_proj = st.text_input("Project Name", value="Finance Reporting Bot")
            new_rpm = st.number_input("Rate Limit (RPM)", value=60)
            if st.form_submit_button("Create Key"):
                with st.session_state.db._get_connection() as conn:
                    conn.execute("INSERT OR REPLACE INTO virtual_keys (key, project_name, rate_limit_rpm, created_at, is_active) VALUES (?, ?, ?, datetime('now'), 1)", (new_key, new_proj, new_rpm))
                    conn.commit()
                st.success(f"Key '{new_key}' created!")
                st.rerun()

    with st.session_state.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, project_name, rate_limit_rpm, created_at, is_active FROM virtual_keys")
        keys_df = pd.DataFrame([dict(r) for r in cursor.fetchall()])
        st.dataframe(keys_df, use_container_width=True)

with tab_arch:
    st.markdown("""
    ### Gateway Middleware Pipeline
    1. **Authentication**: Resolves incoming Bearer token to virtual keys in SQLite.
    2. **Rate Limiting**: Enforces token-bucket sliding-window rate limit per key.
    3. **Exact Cache**: Matches SHA-256 of `(model, messages, temperature)` to instantly return cached payloads at $0 cost and <5ms latency.
    4. **Smart Routing & Fallback**: Routes to requested provider; if unreachable or rate-limited, cascades down the fallback priority chain.
    5. **Usage & Cost Tracking**: Computes token cost and logs complete audit traces into SQLite.
    """)
