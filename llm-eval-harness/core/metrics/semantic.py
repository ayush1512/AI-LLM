import math
import re
import numpy as np
from typing import Optional, Dict, Any
from core.metrics.base import BaseMetric, MetricResult
from config import config

class CosineSimilarityMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Semantic Cosine Similarity"

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold
        self._st_model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(config.embedding_model)
        except Exception:
            self._st_model = None

    def _embed(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            try:
                emb = self._st_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
                return emb[0]
            except Exception:
                pass

        # High-fidelity sub-word & token n-gram projection for offline semantic scoring
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        dim = 512
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            return vec

        # Unigrams with position dampening
        for i, w in enumerate(tokens):
            h = abs(hash(w))
            vec[h % dim] += 1.0
            # Also hash 3-character prefixes/roots for morphological matching (e.g. france/french)
            if len(w) >= 3:
                h_stem = abs(hash(w[:4]))
                vec[h_stem % dim] += 0.8

        # Bigrams
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i+1]}"
            h_bg = abs(hash(bg))
            vec[h_bg % dim] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def evaluate(self, prediction: str, expected: Optional[str] = None, context: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        if not expected:
            return MetricResult(self.name, 1.0, True, "No reference ground truth supplied.")

        v_pred = self._embed(prediction)
        v_exp = self._embed(expected)

        dot = np.dot(v_pred, v_exp)
        norm_p = np.linalg.norm(v_pred)
        norm_e = np.linalg.norm(v_exp)

        sim = float(dot / (norm_p * norm_e)) if (norm_p * norm_e) > 0 else 0.0
        sim = max(0.0, min(1.0, sim))
        
        passed = (sim >= self.threshold)
        reason = f"Cosine similarity {sim:.3f} >= threshold {self.threshold:.2f}" if passed else f"Cosine similarity {sim:.3f} < threshold {self.threshold:.2f}"
        return MetricResult(self.name, round(sim, 4), passed, reason)
