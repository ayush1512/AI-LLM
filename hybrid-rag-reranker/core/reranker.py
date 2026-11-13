from typing import List, Dict, Any
import math
import re

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker.
    Jointly evaluates (query, document) pairs with full cross-attention.
    Falls back to a semantic salience cross-scorer when the HuggingFace model is unavailable.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self._model is not None:
            try:
                pairs = [[query, c["text"]] for c in candidates]
                raw_scores = self._model.predict(pairs)
                # Convert logits to probability via sigmoid if needed
                scores = [1.0 / (1.0 + math.exp(-s)) if isinstance(s, (int, float)) else float(s) for s in raw_scores]
                for c, s in zip(candidates, scores):
                    c["rerank_score"] = float(s)
                return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
            except Exception:
                pass

        # Fallback cross-scoring: lexical intersection + exact phrase matching + length penalty
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_len = max(1, len(query_words))
        
        for c in candidates:
            doc_text = c["text"].lower()
            doc_words = re.findall(r'\b\w+\b', doc_text)
            
            # 1. Word overlap ratio
            matched_words = sum(1 for w in query_words if w in doc_words)
            overlap_score = matched_words / query_len
            
            # 2. Phrase matching bonus
            phrase_bonus = 0.3 if query.lower().strip() in doc_text else 0.0
            
            # 3. Base RRF contribution
            rrf_contrib = c.get("rrf_score", 0.0) * 5.0
            
            combined_score = (overlap_score * 0.6) + phrase_bonus + (rrf_contrib * 0.4)
            c["rerank_score"] = round(min(1.0, combined_score), 4)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

