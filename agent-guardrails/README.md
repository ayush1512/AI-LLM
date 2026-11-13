# 🛡️ Enterprise ReAct Agent with Layered Guardrails

An autonomous ReAct tool-calling agent wrapped with multi-layered, zero-trust safety guardrails inspired by **NeMo Guardrails and OWASP LLM Top 10**.

---

## 🏗️ Architecture

```
User Prompt ──► [ Input Guardrails ] ─────────────┐
                ├─ Prompt Injection Detector       │ (Blocked: Instant Rejection)
                ├─ Topic Boundary Enforcement     ▼
                └─ PII Detection & Redaction ──► [ ReAct Agent Loop ]
                                                        │
                                                        ▼ (Tool Call Required)
                                                [ Tool Guardrails ]
                                                ├─ Whitelist Auth
                                                └─ Param Sanitization
                                                        │
                                                        ▼
                                                [ Execute Secure Tool ]
                                                        │
                                                        ▼
                                                [ Synthesize LLM Output ]
                                                        │
                                                        ▼
                                                [ Output Guardrails ]
                                                ├─ Toxicity Filter
                                                ├─ PII Leak Prevention
                                                └─ Groundedness Check
                                                        │
                                                        ▼
                                                Final Safe Response
```

---

## 🚀 Quickstart

### 1. Installation
```bash
cd agent-guardrails
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your OpenAI API credentials (optional — offline simulation runs automatically if no key is supplied):
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
python tests/test_guardrails.py
```

---

## 📁 Project Structure

```
agent-guardrails/
├── app.py                     # Streamlit Chat & Audit Telemetry UI
├── config.py                  # Agent & Guardrail configurations
├── requirements.txt           # Pinned dependencies
├── .env.example               # Environment variables template
├── README.md                  # Documentation
├── core/
│   ├── __init__.py
│   ├── agent.py               # Guarded ReAct Agent loop
│   ├── tools.py               # Whitelisted enterprise tools & schemas
│   └── guardrails/
│       ├── __init__.py
│       ├── base.py            # BaseRail and RailResult dataclass
│       ├── input_rails.py     # PII, prompt injection, and topic boundaries
│       ├── tool_rails.py      # Authorization whitelist & argument validator
│       ├── output_rails.py    # Toxicity, PII leak, and groundedness checks
│       └── pipeline.py        # Central multi-tier guardrail pipeline
└── tests/
    └── test_guardrails.py     # Unit test suite
```

---

## 🛡️ Guardrail Specifications
- **Prompt Injection Defense**: Intercepts jailbreaks, DAN prompts, instruction resets, and system prompt leakage attempts.
- **Topic Boundaries**: Keeps the agent strictly focused on business operations, finance, and company policies, discarding off-topic discussions.
- **PII Detection**: Automatically redacts emails/phone numbers and halts transmission of high-risk SSNs and credit card numbers.
- **Tool Sandbox**: Restricts function calls to whitelisted APIs with mathematical and string safety filters.
- **Output Leak Prevention**: Guarantees no sensitive data or abusive language reaches the end user.

