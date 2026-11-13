from typing import List, Dict, Any
import numpy as np
from core.chunker import TextChunk
from core.embeddings import EmbeddingModel

class DenseRetriever:
    """
    Qdrant-backed dense vector retriever.
    Supports in-memory mode (:memory:) and remote Qdrant instances.
    Provides an in-memory cosine store fallback when qdrant-client is absent.
    """
    def __init__(self, embedding_model: EmbeddingModel, collection_name: str = "hybrid_rag", location: str = ":memory:"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.location = location
        self.client = None
        self._fallback_vectors: List[np.ndarray] = []
        self._fallback_payloads: List[Dict[str, Any]] = []
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self.client = QdrantClient(location=self.location)
            # Re-create collection for clean indexing
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_model.dim, distance=Distance.COSINE)
            )
        except Exception:
            self.client = None

    def index_chunks(self, chunks: List[TextChunk]):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_texts(texts)

        if self.client is not None:
            try:
                from qdrant_client.models import PointStruct
                points = [
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vectors[i].tolist(),
                        payload={
                            "text": chunk.text,
                            "source": chunk.source,
                            "word_count": chunk.word_count,
                            "chunk_id": chunk.chunk_id
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
                return
            except Exception:
                pass

        # Fallback in-memory index
        self._fallback_vectors = [v for v in vectors]
        self._fallback_payloads = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "word_count": chunk.word_count,
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_query(query)

        if self.client is not None:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k
                )
                return [
                    {
                        "chunk_id": hit.payload["chunk_id"],
                        "text": hit.payload["text"],
                        "source": hit.payload["source"],
                        "score": float(hit.score),
                        "retrieval_type": "dense"
                    }
                    for hit in hits
                ]
            except Exception:
                pass

        # Fallback cosine search
        if not self._fallback_vectors:
            return []

        scores = []
        for i, vec in enumerate(self._fallback_vectors):
            dot = np.dot(query_vector, vec)
            norm_q = np.linalg.norm(query_vector)
            norm_v = np.linalg.norm(vec)
            sim = dot / (norm_q * norm_v) if (norm_q * norm_v) > 0 else 0.0
            scores.append((float(sim), i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, idx in scores[:top_k]:
            payload = self._fallback_payloads[idx]
            results.append({
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "source": payload["source"],
                "score": sim,
                "retrieval_type": "dense"
            })
        return results

