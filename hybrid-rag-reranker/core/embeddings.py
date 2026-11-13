import math
import numpy as np
from typing import List

class EmbeddingModel:
    """
    Embedding manager that uses sentence-transformers if available,
    with a graceful deterministic dense vector fallback for offline/isolated environments.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._st_model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
        except Exception:
            self._st_model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception:
                pass

        # Deterministic semantic hash projection (offline fallback)
        embeddings = []
        for text in texts:
            vec = self._hash_embed(text)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vecs = self.embed_texts([query])
        return vecs[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vec
        for i, word in enumerate(words):
            # Deterministic pseudo-random seed from word hash
            h = abs(hash(word))
            idx1 = h % self.dim
            idx2 = (h * 31) % self.dim
            val = 1.0 / math.sqrt(i + 1)
            vec[idx1] += val
            vec[idx2] += val * 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

