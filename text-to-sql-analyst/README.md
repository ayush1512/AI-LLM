# 📊 Enterprise Text-to-SQL Analyst

A production-grade, natural language database analytics system featuring **dynamic schema retrieval, few-shot golden repositories, AST safety validation, and an agentic self-correction repair loop**.

---

## 🏗️ Architecture

```
User Natural Language Query
          │
          ▼
 [ Dynamic Schema Scoping ] (Extracts relevant tables only, avoiding context dilution)
          │
          ▼
 [ Agentic SQL Generation ] (Prompt injected with Schema + Few-shot Golden Pairs)
          │
          ▼
 [ AST & Safety Validation ] ──► (Violations / DROP / DELETE) ──► Blocked
          │
          ▼ (Safe SELECT with enforced LIMIT)
 [ Read-Only SQLite Sandbox ] (file:...mode=ro)
          │
     ┌────┴──────────────────────────┐
     ▼                               ▼
 [ Operational/Syntax Error ]   [ Execution Success ]
     │                               │
     ▼                               ▼
 [ Self-Correction Loop ]       [ Result Table & Auto-Charts ]
 (Refeeds error to LLM, <= 3)    (Plotly Bar/Pie/Line rendering)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
cd text-to-sql-analyst
pip install -r requirements.txt
```

### 2. Configuration & Database Setup
Seed the e-commerce analytics SQLite database:
```bash
python data/seed_db.py
```
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your OpenAI-compatible API credentials (optional — offline simulation answers standard analytical requests out-of-the-box):
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

### 3. Run the Streamlit Interface
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
python tests/test_text_to_sql.py
```

---

## 📁 Project Structure

```
text-to-sql-analyst/
├── app.py                     # Streamlit Analytics & Plotly Visualization UI
├── config.py                  # Database & Model configuration
├── requirements.txt           # Pinned production dependencies
├── .env.example               # Environment variables template
├── README.md                  # Documentation
├── data/
│   ├── seed_db.py             # Database creation & seeding script
│   └── few_shot_queries.json  # Verified golden NL-to-SQL prompt bank
├── core/
│   ├── __init__.py
│   ├── schema.py              # Semantic layer & table intent scoping
│   ├── validator.py           # SQL AST parser, keyword blocker, & LIMIT enforcer
│   └── engine.py              # Self-correction loop & read-only execution engine
└── tests/
    └── test_text_to_sql.py    # Unit tests
```

---

## 🔒 Security & Governance
- **Strict Read-Only Mode**: SQLite URI opened with `mode=ro` preventing any disk modification.
- **AST Destructive Operation Blocker**: Forbids `DROP`, `DELETE`, `UPDATE`, `ALTER`, `INSERT`, `TRUNCATE`, and `PRAGMA`.
- **Automatic Limit Enforcement**: Injects a strict safety `LIMIT` clause on all unbounded queries to protect against runaway table scans.
- **Agentic Self-Correction**: Automatically catches SQL column errors or join typos, re-prompting the LLM with error details up to 3 times before failing gracefully.

