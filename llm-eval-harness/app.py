import streamlit as st
import pandas as pd
import json
import plotly.express as px
from core.dataset import load_dataset_from_jsonl, EvalCase
from core.metrics.deterministic import (
    ExactMatchMetric, KeywordPresenceMetric, JsonSchemaMetric, LengthBoundMetric
)
from core.metrics.semantic import CosineSimilarityMetric
from core.metrics.llm_judge import get_default_judges
from core.runner import EvaluationRunner
from config import config

st.set_page_config(
    page_title="LLM Eval Harness",
    page_icon="⚖️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-pass {
        color: #2e7d32;
        font-weight: bold;
    }
    .metric-fail {
        color: #c62828;
        font-weight: bold;
    }
    .card-detail {
        background-color: #f9f9f9;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #1976d2;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

if "eval_report" not in st.session_state:
    st.session_state.eval_report = None

st.title("⚖️ LLM Evaluation Harness")
st.caption("Custom multi-tier eval framework: Deterministic checks, Semantic Similarity, & LLM-as-a-Judge (G-Eval).")

# Sidebar - Dataset & Configuration
with st.sidebar:
    st.header("⚙️ Evaluation Setup")
    user_api_key = st.text_input("OpenAI API Key", value=config.openai_api_key, type="password")
    model_under_test = st.text_input("Model Under Test", value=config.model_under_test)
    judge_model = st.text_input("Judge Model", value=config.judge_model)

    st.divider()
    st.subheader("📁 Test Suite")
    suite_source = st.radio("Choose Dataset", ["Default Benchmark Suite", "Upload Custom JSONL"])
    
    cases_loaded: list[EvalCase] = []
    if suite_source == "Default Benchmark Suite":
        cases_loaded = load_dataset_from_jsonl("data/benchmark_suite.jsonl")
        st.info(f"Loaded **{len(cases_loaded)}** benchmark test cases.")
    else:
        uploaded_file = st.file_uploader("Upload JSONL test cases", type=["jsonl", "json"])
        if uploaded_file:
            cases_loaded = load_dataset_from_jsonl(uploaded_file.read())
            st.success(f"Loaded {len(cases_loaded)} cases from file.")

    st.divider()
    st.subheader("🎯 Active Metrics")
    m_keyword = st.checkbox("Keyword Presence & Constraints", value=True)
    m_json = st.checkbox("JSON Schema Verification", value=True)
    m_length = st.checkbox("Length & Verbosity Bounds", value=True)
    m_semantic = st.checkbox("Semantic Cosine Similarity", value=True)
    m_judges = st.checkbox("LLM-as-a-Judge (Faithfulness/Relevancy)", value=True)

    active_metrics = []
    if m_keyword:
        active_metrics.append(KeywordPresenceMetric())
    if m_json:
        active_metrics.append(JsonSchemaMetric())
    if m_length:
        active_metrics.append(LengthBoundMetric())
    if m_semantic:
        active_metrics.append(CosineSimilarityMetric())
    if m_judges:
        active_metrics.extend(get_default_judges())

# Main Run Action
col_action1, col_action2 = st.columns([2, 1])
with col_action1:
    run_btn = st.button("🚀 Run Evaluation Suite", type="primary", use_container_width=True)

if run_btn:
    if not cases_loaded:
        st.error("No test cases available. Load benchmark or upload a JSONL file.")
    elif not active_metrics:
        st.error("Select at least one metric to evaluate.")
    else:
        runner = EvaluationRunner(metrics=active_metrics)
        progress_bar = st.progress(0.0)
        
        with st.spinner("Executing model generations and evaluating metrics..."):
            report = runner.run_suite(
                cases=cases_loaded,
                model=model_under_test,
                custom_api_key=user_api_key,
                progress_callback=progress_bar.progress
            )
            st.session_state.eval_report = report
            progress_bar.empty()
            st.success(f"Evaluation complete in {report['total_duration_sec']} seconds!")

# Display Results Dashboard
if st.session_state.eval_report:
    report = st.session_state.eval_report
    
    st.divider()
    st.subheader("📊 Executive Evaluation Scorecard")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Pass Rate", f"{report['overall_pass_rate_pct']}%")
    c2.metric("Cases Evaluated", report["cases_evaluated"])
    c3.metric("Duration", f"{report['total_duration_sec']}s")
    c4.metric("Active Metrics", len(report["scorecard"]))

    # Metric Breakdown Bar Chart
    scorecard_data = []
    for m_name, stats in report["scorecard"].items():
        scorecard_data.append({
            "Metric": m_name,
            "Average Score": stats["avg_score"] * 100,
            "Pass Rate (%)": stats["pass_rate_pct"],
            "Passed": stats["passes"],
            "Total": stats["total"]
        })
    df_scores = pd.DataFrame(scorecard_data)

    col_chart, col_score_tbl = st.columns([1, 1])
    with col_chart:
        fig = px.bar(
            df_scores,
            x="Metric",
            y="Pass Rate (%)",
            color="Pass Rate (%)",
            color_continuous_scale="RdYlGn",
            range_y=[0, 100],
            title="Pass Rate % by Evaluation Metric"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_score_tbl:
        st.write("**Metric Performance Table**")
        st.dataframe(df_scores, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🔍 Case-by-Case Deep Dive")
    for case in report["results"]:
        status_icon = "🟢 PASS" if case["passed"] else "🔴 FAIL"
        with st.expander(f"{status_icon} | Case ID: `{case['case_id']}` [{case['task_type'].upper()}] — Score: {case['overall_score']:.2f}"):
            st.write(f"**Prompt:** {case['input_prompt']}")
            if case.get("context"):
                st.caption(f"**Context:** {case['context']}")
            
            col_p, col_e = st.columns(2)
            with col_p:
                st.markdown("**Model Prediction:**")
                st.info(case["prediction"])
            with col_e:
                st.markdown("**Expected Ground Truth:**")
                st.code(case.get("expected_output") or "None specified", language="text")

            st.write("**Metric Scoring Details:**")
            for m in case["metric_details"]:
                res_badge = "✅" if m["passed"] else "❌"
                st.markdown(f"- {res_badge} **{m['metric']}** (Score: `{m['score']:.2f}`) — {m['reason']}")

    # Export Report
    st.divider()
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        json_report = json.dumps(report, indent=2)
        st.download_button(
            label="💾 Download Evaluation Report (JSON)",
            data=json_report,
            file_name="eval_report.json",
            mime="application/json",
            use_container_width=True
        )
    with col_exp2:
        df_flat = pd.DataFrame([
            {
                "case_id": c["case_id"],
                "task_type": c["task_type"],
                "input_prompt": c["input_prompt"],
                "prediction": c["prediction"],
                "passed": c["passed"],
                "overall_score": c["overall_score"],
                "latency_ms": c["latency_ms"]
            }
            for c in report["results"]
        ])
        st.download_button(
            label="📊 Download CSV Summary",
            data=df_flat.to_csv(index=False),
            file_name="eval_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

