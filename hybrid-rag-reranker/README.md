# ⚡ HybridRAG + Cross-Encoder Reranking

An industry-grade, production-ready Retrieval-Augmented Generation (RAG) system combining **Dense Semantic Vector Search (Qdrant)** with **Sparse Lexical Search (BM25)**, unified via **Reciprocal Rank Fusion (RRF)**, and refined with a **Cross-Encoder Reranker**.

---

## 🏗️ Architecture

```
User Query ──► [ Dense Retrieval (Qdrant ANN) ] ──┐
             └──► [ Sparse Retrieval (BM25)     ] ──┼──► [ Reciprocal Rank Fusion (RRF) ]
                                                     │               │
                                                     │         Top Candidates (~30)
                                                     │               │
                                                     │               ▼
                                                     └────► [ Cross-Encoder Reranker ]
                                                                     │
                                                              Top 5 Contexts
                                                                     │
                                                                     ▼
                                                          [ LLM Answer Generation ]
```

### Why This Architecture?
1. **Dense Retrieval alone fails** on exact keywords, part numbers, version strings, error traces, and jargon.
2. **Sparse Retrieval (BM25) alone fails** on synonyms, paraphrasing, and semantic queries.
3. **Reciprocal Rank Fusion (RRF)** solves the score-calibration problem by fusing ranks rather than disparate scores:
   $$RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$$
4. **Cross-Encoder Reranking** scores `(query, document)` jointly through deep cross-attention layers, yielding a 30–45% lift in top-k precision.

---

## 🚀 Quickstart

### 1. Installation
```bash
cd hybrid-rag-reranker
pip install -r requirements.txt
```

### 2. Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Fill in your configuration:
```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
QDRANT_LOCATION=:memory:
```

### 3. Run the Streamlit UI
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
python -m unittest discover tests
```

---

## 📁 Project Structure

```
hybrid-rag-reranker/
├── app.py                     # Streamlit Minimalistic UI
├── config.py                  # Pydantic & environment settings
├── requirements.txt           # Pinned production dependencies
├── .env.example               # Template environment configuration
├── README.md                  # System documentation
├── core/
│   ├── __init__.py
│   ├── chunker.py             # Overlapping text chunker & PDF parser
│   ├── embeddings.py          # Embedding manager (sentence-transformers + fallback)
│   ├── dense_retriever.py     # Qdrant client integration
│   ├── sparse_retriever.py    # BM25 lexical retriever
│   ├── fusion.py              # Reciprocal Rank Fusion (RRF) algorithm
│   ├── reranker.py            # Cross-Encoder joint scoring
│   └── rag_engine.py          # Unified pipeline orchestrator
└── tests/
    └── test_hybrid_rag.py     # Comprehensive unit tests
```

---

## ⚙️ Features
- **Dense Vector Search**: Qdrant vector database (`:memory:` for development, or remote cluster).
- **Sparse Lexical Search**: BM25 Okapi tokenizer with length normalization.
- **RRF Rank Merging**: Parameterized $k=60$ fusion.
- **Cross-Encoder Reranker**: Joint query-chunk attention model.
- **Minimalist Streamlit UI**: Document upload, latency diagnostics, and side-by-side before/after reranking inspection.

