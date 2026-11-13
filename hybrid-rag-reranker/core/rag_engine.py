from typing import List, Dict, Any, Optional
import time
from core.chunker import chunk_text, extract_text_from_file, TextChunk
from core.embeddings import EmbeddingModel
from core.dense_retriever import DenseRetriever
from core.sparse_retriever import SparseRetriever
from core.fusion import reciprocal_rank_fusion
from core.reranker import CrossEncoderReranker
from config import settings

class HybridRAGEngine:
    """
    End-to-end Hybrid RAG Engine with:
    - Dense Vector Search (Qdrant)
    - Sparse BM25 Keyword Search
    - Reciprocal Rank Fusion (RRF)
    - Cross-Encoder Reranking
    - LLM Generation (OpenAI-compatible)
    """
    def __init__(self):
        self.embedding_model = EmbeddingModel(model_name=settings.embedding_model)
        self.dense_retriever = DenseRetriever(
            embedding_model=self.embedding_model,
            location=settings.qdrant_location
        )
        self.sparse_retriever = SparseRetriever()
        self.reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        self.indexed_chunks: List[TextChunk] = []

    def index_document(self, content: str, source_name: str = "document"):
        chunks = chunk_text(
            text=content,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            source=source_name
        )
        self.indexed_chunks.extend(chunks)
        # Re-assign sequential IDs
        for i, c in enumerate(self.indexed_chunks):
            c.chunk_id = i

        self.dense_retriever.index_chunks(self.indexed_chunks)
        self.sparse_retriever.index_chunks(self.indexed_chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k_retrieval: int = None, top_k_rerank: int = None) -> Dict[str, Any]:
        start_time = time.time()
        k_retrieve = top_k_retrieval or settings.top_k_retrieval
        k_rerank = top_k_rerank or settings.top_k_rerank

        t0 = time.time()
        dense_results = self.dense_retriever.search(query, top_k=k_retrieve)
        dense_latency = (time.time() - t0) * 1000

        t1 = time.time()
        sparse_results = self.sparse_retriever.search(query, top_k=k_retrieve)
        sparse_latency = (time.time() - t1) * 1000

        t2 = time.time()
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=settings.rrf_k,
            top_k=k_retrieve
        )
        fusion_latency = (time.time() - t2) * 1000

        t3 = time.time()
        reranked_results = self.reranker.rerank(query, [c.copy() for c in fused_candidates], top_k=k_rerank)
        rerank_latency = (time.time() - t3) * 1000

        total_latency = (time.time() - start_time) * 1000

        return {
            "query": query,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused_candidates[:k_rerank],
            "reranked_results": reranked_results,
            "latencies_ms": {
                "dense": round(dense_latency, 2),
                "sparse": round(sparse_latency, 2),
                "fusion": round(fusion_latency, 2),
                "rerank": round(rerank_latency, 2),
                "total": round(total_latency, 2)
            }
        }

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], custom_api_key: Optional[str] = None) -> str:
        api_key = custom_api_key or settings.openai_api_key
        if not api_key:
            # Informative simulated response if API key is not configured
            context_summary = "\n".join([f"- [{c['source']} #chunk-{c['chunk_id']}]: {c['text'][:150]}..." for c in context_chunks])
            return (
                f"### [Offline Simulation Mode - OPENAI_API_KEY not configured]\n\n"
                f"**Based on the top retrieved context chunks:**\n{context_summary}\n\n"
                f"**Answer Summary:**\n"
                f"The system successfully retrieved the relevant evidence for '{query}'. Configure your `OPENAI_API_KEY` in `.env` to enable full LLM generative syntheses."
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=settings.openai_base_url)
            
            context_text = "\n\n".join([
                f"[Source: {c['source']} | Chunk ID: {c['chunk_id']} | Relevance: {c.get('rerank_score', c.get('rrf_score', 0)):.4f}]\n{c['text']}"
                for c in context_chunks
            ])

            system_prompt = (
                "You are an expert technical AI assistant using a Hybrid RAG + Cross-Encoder Reranking system. "
                "Synthesize a clear, accurate, and comprehensive answer strictly grounded in the provided context. "
                "Always cite your sources using [Source: filename | Chunk ID: #] format."
            )

            user_prompt = f"Context:\n{context_text}\n\nUser Question:\n{query}\n\nAnswer:"

            response = client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error contacting OpenAI-compatible endpoint: {str(e)}"

