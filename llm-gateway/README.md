# 🛡️ OpenAI-Compatible LLM Gateway

A high-performance, unified API gateway proxy for Large Language Models. Centralizes routing to **OpenAI, Anthropic Claude, Google Gemini, Ollama, and local models** behind a single OpenAI-compatible `/v1/chat/completions` endpoint.

---

## 🏗️ Architecture

```
Client App / SDK ──► [ OpenAI /v1/chat/completions ]
                              │
                              ▼
           ┌─────────────────────────────────────┐
           │     LLM Gateway Middleware Stack    │
           │  ├─ Virtual API Key Authentication  │
           │  ├─ Token-Bucket Rate Limiter       │
           │  ├─ SHA-256 Exact Response Cache    │
           │  ├─ Smart Fallback Router           │
           │  └─ SQLite Usage & Cost Logger      │
           └─────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    [ OpenAI API ]     [ Anthropic API ]  [ Google Gemini ]
    (gpt-4o, mini)     (Claude 3.5 Sonnet) (Gemini 1.5 Flash)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
cd llm-gateway
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your upstream provider API keys (optional — the gateway ships with a deterministic `mock` provider for testing without API keys):
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

### 3. Run the Observability Dashboard
```bash
streamlit run app.py
```

### 4. Run the FastAPI Proxy Server (Optional)
```bash
uvicorn gateway.main:app --port 8000 --reload
```

Then send standard OpenAI client requests:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="gw-dev-test-key-123"  # Virtual Gateway Key
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain hybrid search."}]
)
print(response.choices[0].message.content)
```

### 5. Run Automated Tests
```bash
python tests/test_gateway.py
```

---

## 📁 Project Structure

```
llm-gateway/
├── app.py                     # Streamlit Observability Dashboard
├── config.py                  # Gateway configuration
├── requirements.txt           # Production dependencies
├── .env.example               # Environment variables template
├── README.md                  # Documentation
├── gateway/
│   ├── __init__.py
│   ├── main.py                # FastAPI proxy server (/v1/chat/completions)
│   ├── router.py              # Central router with fallback orchestration
│   ├── db.py                  # SQLite usage, telemetry, and key store
│   ├── middleware.py          # Rate limiting, caching, and cost calculator
│   └── providers/
│       ├── __init__.py
│       ├── base.py            # LLMProvider abstract base class
│       ├── mock_provider.py   # High-speed deterministic mock provider
│       ├── openai_provider.py # OpenAI native adapter
│       ├── anthropic_provider.py # Claude Messages adapter
│       ├── gemini_provider.py # Google Gemini adapter
│       └── ollama_provider.py # Local Ollama adapter
└── tests/
    └── test_gateway.py        # Comprehensive test suite
```

---

## ⚡ Core Features
- **Unified OpenAI Standard**: Any tool or agent supporting OpenAI can plug in.
- **Virtual API Keys**: Isolate applications and teams with granular RPM rate limits.
- **Exact-Match Response Cache**: Sub-millisecond replies on duplicate requests with zero token cost.
- **Provider Fallback**: Seamlessly fails over to secondary providers during upstream outages or rate limits.
- **Live Observability**: Real-time token consumption, latency metrics, and spend tracking.

