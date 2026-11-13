# ⚖️ Production LLM Evaluation Harness

A comprehensive, custom evaluation harness and benchmarking framework for Large Language Models. Supports **Deterministic checks, Semantic Cosine Similarity, and Chain-of-Thought LLM-as-a-Judge metrics (G-Eval)** with batch test execution and a Streamlit analytics dashboard.

---

## 🏗️ Architecture

```
[ Evaluation Dataset (.jsonl) ]
               │
               ▼
   [ Batch Evaluation Runner ] ──► [ Model Under Test ] ──► Predictions
               │
               ▼
     ┌──────────────────────────────────────────────────┐
     │               Modular Metric Pipeline            │
     │  ├─ Deterministic: ExactMatch, Keywords, JSON    │
     │  ├─ Semantic: Cosine Similarity (Embeddings)     │
     │  └─ LLM-as-a-Judge: Faithfulness, Relevancy,     │
     │                     Conciseness (G-Eval Rubric)  │
     └──────────────────────────────────────────────────┘
               │
               ▼
     [ Executive Scorecard & Drill-Down Dashboard ]
     (Pass Rate %, Latency, Score Distribution, CSV/JSON Exports)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
cd llm-eval-harness
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your OpenAI-compatible API key (optional — includes deterministic offline simulation for offline test suite verification):
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_UNDER_TEST=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
```

### 3. Run the Evaluation Dashboard
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
python tests/test_eval_harness.py
```

---

## 📁 Project Structure

```
llm-eval-harness/
├── app.py                     # Streamlit Executive Dashboard & Inspector
├── config.py                  # Evaluation parameters and model settings
├── requirements.txt           # Pinned production dependencies
├── .env.example               # Environment variables template
├── README.md                  # Documentation
├── data/
│   └── benchmark_suite.jsonl  # Pre-built benchmark covering QA, JSON, and safety
├── core/
│   ├── __init__.py
│   ├── dataset.py             # EvalCase dataclass and JSONL dataset loader
│   ├── runner.py              # Batch runner, progress callback, & score compiler
│   └── metrics/
│       ├── __init__.py
│       ├── base.py            # MetricResult and BaseMetric abstraction
│       ├── deterministic.py   # Exact match, keyword presence, JSON schema, length
│       ├── semantic.py        # Embeddings-based Cosine Similarity
│       └── llm_judge.py       # Chain-of-Thought LLM-as-a-Judge (G-Eval style)
└── tests/
    └── test_eval_harness.py   # Unit test suite
```

---

## 📊 Evaluation Tiers
1. **Tier 1 (Deterministic)**: Fast, $0 cost, reproducible checks for JSON format, keyword presence, forbidden tokens, and word length bounds.
2. **Tier 2 (Semantic Similarity)**: Measures conceptual proximity without requiring exact wording using cosine vector distance.
3. **Tier 3 (LLM-as-a-Judge)**: Chain-of-thought rubric scoring on a 1–5 scale evaluating:
   - **Faithfulness**: Verifies claims against provided context (hallucination detection).
   - **Answer Relevancy**: Assesses directness and query addressing.
   - **Conciseness**: Evaluates efficiency and lack of padding.

