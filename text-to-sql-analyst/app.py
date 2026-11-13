import streamlit as st
import pandas as pd
import plotly.express as px
from core.engine import TextToSQLEngine
from core.schema import SCHEMA_METADATA
from config import config

st.set_page_config(
    page_title="Text-to-SQL Analyst",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .sql-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 12px;
        border-radius: 6px;
        font-family: monospace;
    }
    .badge-ro {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = TextToSQLEngine()

st.title("📊 Enterprise Text-to-SQL Analyst")
st.caption("Natural language analytics with schema scoping, AST safety validation, & self-correction retry loops.")

# Sidebar - Schema & Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("OpenAI API Key", value=config.openai_api_key, type="password")
    
    st.markdown('<span class="badge-ro">🔒 SQLite Connection: READ-ONLY</span>', unsafe_allow_html=True)
    st.write(f"**Database:** `{config.database_path}`")
    
    st.divider()
    st.subheader("🗄️ Database Schema Explorer")
    for table_name, meta in SCHEMA_METADATA.items():
        with st.expander(f"📁 {table_name}"):
            st.caption(meta["description"])
            cols_df = pd.DataFrame([
                {"Column": col, "Description & Type": desc}
                for col, desc in meta["columns"].items()
            ])
            st.dataframe(cols_df, hide_index=True, use_container_width=True)

# Sample Query Shortcuts
st.subheader("💡 Recommended Analysis Prompts")
c1, c2, c3, c4 = st.columns(4)
query_to_run = ""
if c1.button("🏆 Top 5 Customers by Spend", use_container_width=True):
    query_to_run = "Show top 5 customers by total order amount"
if c2.button("📦 Category Revenue Breakdown", use_container_width=True):
    query_to_run = "What is the total revenue for each product category?"
if c3.button("⚠️ Low Stock Inventory (< 50)", use_container_width=True):
    query_to_run = "List products that have low stock (less than 50 units)"
if c4.button("🏙️ Orders by City", use_container_width=True):
    query_to_run = "How many orders were placed by customers in each city?"

user_query = st.text_input("Enter natural language business question:", value=query_to_run, placeholder="e.g. Which customers spent over $300 in completed orders?")

if user_query:
    with st.spinner("Analyzing schema, compiling SQL, and verifying safety constraints..."):
        result = st.session_state.engine.run_analyst_pipeline(
            question=user_query,
            custom_api_key=user_api_key
        )

    tab_results, tab_sql, tab_self_correct, tab_schema = st.tabs([
        "📈 Query Results & Visuals",
        "💻 Generated SQL",
        "🔄 Self-Correction Log",
        "📑 Scoped Schema"
    ])

    with tab_results:
        if result["success"]:
            st.success(f"Query returned {result['row_count']} rows in {result['total_latency_ms']} ms.")
            df = pd.DataFrame(result["data"])
            
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                st.subheader("Data Table")
                st.dataframe(df, use_container_width=True)

            with col_t2:
                st.subheader("Automated Chart")
                # Auto-visualize if suitable columns exist
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()

                if len(numeric_cols) >= 1 and len(text_cols) >= 1:
                    chart_type = st.radio("Chart Type", ["Bar", "Pie", "Line"], horizontal=True)
                    x_col = text_cols[0]
                    y_col = numeric_cols[0]

                    if chart_type == "Bar":
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", color_discrete_sequence=['#1f77b4'])
                    elif chart_type == "Pie":
                        fig = px.pie(df, names=x_col, values=y_col, title=f"Share of {y_col} by {x_col}")
                    else:
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} trend", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Single column result; tabular display is optimal.")
        else:
            st.error("Query execution could not be satisfied within maximum self-correction limits.")
            st.code(result["final_sql"], language="sql")

    with tab_sql:
        st.subheader("Verified Executable SQL")
        st.code(result["final_sql"], language="sql")
        st.caption("Validated against destructive keywords (`DROP`, `DELETE`, `UPDATE`, `ALTER`) and wrapped in read-only sandbox.")

    with tab_self_correct:
        st.subheader("Agentic Self-Correction Loop")
        st.write(f"Total attempts taken: **{result['total_attempts']}** of {config.max_attempts}")
        for att in result["attempts"]:
            st.markdown(f"**Attempt #{att['attempt']}** — Status: `{att['status']}` ({att['duration_ms']} ms)")
            st.code(att["sql"], language="sql")
            if "error" in att:
                st.warning(f"Error caught: {att['error']}")
            st.divider()

    with tab_schema:
        st.subheader("Dynamically Scoped Tables Passed to LLM Context")
        st.text(result["schema_used"])

