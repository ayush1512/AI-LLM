import math
import re
from typing import List, Dict, Any
from core.chunker import TextChunk

def tokenize(text: str) -> List[str]:
    """Basic lexical tokenizer: lowercases and extracts alphanumeric words."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

class SparseRetriever:
    """
    BM25 lexical sparse retriever.
    Uses rank_bm25 if available or built-in BM25Okapi implementation.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.corpus_size: int = 0
        self._bm25_model = None

    def index_chunks(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self.corpus_size = len(chunks)
        if not chunks:
            return

        tokenized_corpus = [tokenize(c.text) for c in chunks]

        try:
            from rank_bm25 import BM25Okapi
            self._bm25_model = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        except Exception:
            self._bm25_model = None

        # Build native BM25 statistics
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avg_dl = sum(self.doc_len) / max(1, self.corpus_size)

        self.doc_freqs = {}
        for doc in tokenized_corpus:
            seen = set(doc)
            for word in seen:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        if self._bm25_model is not None:
            try:
                scores = self._bm25_model.get_scores(query_tokens)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                results = []
                for idx in ranked_indices:
                    if scores[idx] <= 0:
                        continue
                    c = self.chunks[idx]
                    results.append({
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source": c.source,
                        "score": float(scores[idx]),
                        "retrieval_type": "sparse"
                    })
                return results
            except Exception:
                pass

        # Native BM25 scoring
        scores = []
        for i, c in enumerate(self.chunks):
            doc_tokens = tokenize(c.text)
            dl = len(doc_tokens)
            score = 0.0
            term_counts: Dict[str, int] = {}
            for t in doc_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for token in query_tokens:
                if token not in self.doc_freqs:
                    continue
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs[token]
                # Standard Robertson-Spärck Jones IDF
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / max(0.1, self.avg_dl)))
                score += idf * (numerator / denominator)

            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0.0:
                continue
            c = self.chunks[idx]
            results.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "score": float(score),
                "retrieval_type": "sparse"
            })
        return results

