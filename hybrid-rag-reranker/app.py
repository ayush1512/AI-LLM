import streamlit as st
import time
from core.rag_engine import HybridRAGEngine
from core.chunker import extract_text_from_file
from config import settings

st.set_page_config(
    page_title="HybridRAG + Reranking",
    page_icon="⚡",
    layout="wide"
)

# Custom minimalist styling
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background: #f0f2f6;
        font-size: 0.85rem;
        font-family: monospace;
        margin-right: 6px;
    }
    .chunk-card {
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin-bottom: 10px;
        background: #fafafa;
        border-radius: 0 8px 8px 0;
    }
    .raw-card {
        border-left: 4px solid #2196F3;
        padding: 12px;
        margin-bottom: 10px;
        background: #fafafa;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = HybridRAGEngine()
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

SAMPLE_CORPUS = """
# Production Hybrid RAG & Cross-Encoder Architecture
Hybrid Search combines Dense Semantic Retrieval (bi-encoders) with Sparse Lexical Retrieval (BM25 or SPLADE).
Dense vectors capture conceptual meaning and synonyms but can struggle with exact identifiers, part numbers, and error codes.
Sparse BM25 excels at exact keyword matching and alphanumeric tokens like 'CVE-2024-38077' or 'HTTP 502'.

### Reciprocal Rank Fusion (RRF)
RRF fuses ranked lists without requiring score normalization:
Score(d) = sum_{system} 1 / (60 + rank(d))
This prevents dense vector cosine similarities from skewing against BM25 unbounded scores.

### Cross-Encoder Rerankers
While bi-encoders calculate query and chunk embeddings independently in O(Q + D) time, a cross-encoder performs joint multi-head self-attention across both the query and document tokens simultaneously.
This yields a 30-45% improvement in context precision (MRR@10 and NDCG@10).
Common production rerankers include BGE-Reranker-Large, Cohere Rerank v3, and ms-marco-MiniLM-L-6-v2.
"""

st.title("⚡ HybridRAG + Cross-Encoder Reranker")
st.caption("Dense Vector Search (Qdrant) + Sparse Lexical (BM25) + Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranking")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", value=settings.openai_api_key, type="password")
    model_name = st.text_input("Model Name", value=settings.model_name)
    base_url = st.text_input("Base URL", value=settings.openai_base_url)
    
    st.divider()
    st.subheader("📚 Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDF or TXT", accept_multiple_files=True, type=["pdf", "txt"])
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Index Files", use_container_width=True) and uploaded_files:
            for file in uploaded_files:
                text = extract_text_from_file(file.read(), file.name)
                added = st.session_state.engine.index_document(text, source_name=file.name)
                st.session_state.doc_count += 1
                st.session_state.chunk_count += added
            st.success(f"Indexed {len(uploaded_files)} files!")

    with col_btn2:
        if st.button("Load Sample Doc", use_container_width=True):
            added = st.session_state.engine.index_document(SAMPLE_CORPUS, source_name="architecture_spec.md")
            st.session_state.doc_count += 1
            st.session_state.chunk_count += added
            st.success("Sample doc loaded!")

    st.info(f"**Index Stats:**\n- Documents: {st.session_state.doc_count}\n- Total Chunks: {st.session_state.chunk_count}")

    st.divider()
    st.subheader("🎛️ Retrieval Parameters")
    top_k_retrieval = st.slider("Top-K Candidates (Hybrid)", min_value=5, max_value=50, value=settings.top_k_retrieval)
    top_k_rerank = st.slider("Top-K Reranked Contexts", min_value=1, max_value=10, value=settings.top_k_rerank)

# Main Query Area
query = st.text_input("🔎 Enter your question:", placeholder="e.g. Why does RRF use rank instead of raw scores?")

if query:
    if st.session_state.chunk_count == 0:
        st.warning("⚠️ No documents in index! Please upload a file or click 'Load Sample Doc' in the sidebar.")
    else:
        results = st.session_state.engine.retrieve(query, top_k_retrieval=top_k_retrieval, top_k_rerank=top_k_rerank)
        
        tab_answer, tab_comparison, tab_diag = st.tabs(["✨ Generative Answer", "⚖️ Reranker Comparison", "📊 Latency & Diagnostics"])
        
        with tab_answer:
            with st.spinner("Synthesizing answer from top reranked contexts..."):
                answer = st.session_state.engine.generate_answer(query, results["reranked_results"], custom_api_key=api_key)
            
            st.subheader("Answer")
            st.markdown(answer)
            
            st.divider()
            st.subheader("Top Reranked Contexts Passed to LLM")
            for i, chunk in enumerate(results["reranked_results"], start=1):
                with st.expander(f"Context #{i} | Score: {chunk.get('rerank_score', 0):.4f} | Source: {chunk['source']}"):
                    st.write(chunk["text"])
                    st.caption(f"RRF Rank: {chunk.get('dense_rank', '-')} (dense) / {chunk.get('sparse_rank', '-')} (sparse) | RRF Score: {chunk.get('rrf_score', 0):.4f}")

        with tab_comparison:
            st.subheader("Before vs. After Reranking")
            st.caption("Notice how cross-encoder re-orders candidates to place the most contextually relevant chunks at the top.")
            
            col_pre, col_post = st.columns(2)
            with col_pre:
                st.markdown("#### Top Chunks (Before Rerank - RRF Fusion)")
                for idx, c in enumerate(results["fused_results"], start=1):
                    st.markdown(f"""
                    <div class="raw-card">
                        <strong>#{idx} [{c['source']}]</strong> (RRF Score: {c.get('rrf_score', 0):.4f})<br>
                        <small>{c['text'][:200]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_post:
                st.markdown("#### Top Chunks (After Cross-Encoder Rerank)")
                for idx, c in enumerate(results["reranked_results"], start=1):
                    st.markdown(f"""
                    <div class="chunk-card">
                        <strong>#{idx} [{c['source']}]</strong> (Cross-Encoder: {c.get('rerank_score', 0):.4f})<br>
                        <small>{c['text'][:200]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_diag:
            st.subheader("Pipeline Latency Breakdown")
            lats = results["latencies_ms"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Dense ANN", f"{lats['dense']} ms")
            c2.metric("BM25 Sparse", f"{lats['sparse']} ms")
            c3.metric("RRF Fusion", f"{lats['fusion']} ms")
            c4.metric("Cross-Encoder", f"{lats['rerank']} ms")
            c5.metric("Total Retrieval", f"{lats['total']} ms")
            
            st.divider()
            col_d, col_s = st.columns(2)
            with col_d:
                st.write("**Top 5 Dense Candidates (Qdrant Cosine)**")
                for d in results["dense_results"][:5]:
                    st.code(f"Chunk #{d['chunk_id']} | Cosine: {d['score']:.4f}\n{d['text'][:120]}...", language="text")
            with col_s:
                st.write("**Top 5 Sparse Candidates (BM25 Lexical)**")
                for s in results["sparse_results"][:5]:
                    st.code(f"Chunk #{s['chunk_id']} | BM25: {s['score']:.4f}\n{s['text'][:120]}...", language="text")

