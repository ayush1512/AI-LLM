import unittest
import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunker import chunk_text
from core.embeddings import EmbeddingModel
from core.dense_retriever import DenseRetriever
from core.sparse_retriever import SparseRetriever
from core.fusion import reciprocal_rank_fusion
from core.reranker import CrossEncoderReranker
from core.rag_engine import HybridRAGEngine

class TestHybridRAG(unittest.TestCase):
    def test_chunker(self):
        text = "word " * 120
        chunks = chunk_text(text, chunk_size=50, overlap=10, source="test.txt")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, 0)
        self.assertEqual(chunks[0].source, "test.txt")

    def test_chunker_empty(self):
        chunks = chunk_text("", chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 0)

    def test_dense_retriever(self):
        embedder = EmbeddingModel()
        retriever = DenseRetriever(embedding_model=embedder)
        sample = "PostgreSQL is a relational database. MongoDB is a document database."
        chunks = chunk_text(sample, chunk_size=10, overlap=2)
        retriever.index_chunks(chunks)
        hits = retriever.search("PostgreSQL relational", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("score", hits[0])

    def test_sparse_retriever(self):
        retriever = SparseRetriever()
        text = "Authentication uses JSON Web Tokens (JWT). Database runs on port 5432."
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        retriever.index_chunks(chunks)
        hits = retriever.search("5432", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("5432", hits[0]["text"])

    def test_rrf_fusion(self):
        dense_results = [
            {"chunk_id": 1, "text": "Doc A", "score": 0.95},
            {"chunk_id": 2, "text": "Doc B", "score": 0.80}
        ]
        sparse_results = [
            {"chunk_id": 2, "text": "Doc B", "score": 4.5},
            {"chunk_id": 3, "text": "Doc C", "score": 3.0}
        ]
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=5)
        # Doc 2 is present in both dense (rank 2) and sparse (rank 1), should have highest combined RRF score
        self.assertEqual(fused[0]["chunk_id"], 2)
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_reranker(self):
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": 1, "text": "Apples and bananas are fruits.", "rrf_score": 0.03},
            {"chunk_id": 2, "text": "Vector databases index embeddings for similarity search.", "rrf_score": 0.02}
        ]
        reranked = reranker.rerank("What do vector databases index?", candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["chunk_id"], 2)

    def test_end_to_end_engine(self):
        engine = HybridRAGEngine()
        corpus = (
            "Kubernetes orchestrates containerized workloads across node clusters. "
            "Docker packages applications into standardized container images. "
            "Terraform manages cloud infrastructure as declarative code."
        )
        engine.index_document(corpus, source_name="infra.txt")
        res = engine.retrieve("How does Kubernetes manage containers?", top_k_retrieval=5, top_k_rerank=2)
        self.assertIn("reranked_results", res)
        self.assertGreater(len(res["reranked_results"]), 0)
        self.assertIn("latencies_ms", res)

if __name__ == "__main__":
    unittest.main()
