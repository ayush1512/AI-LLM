from typing import List, Dict, Any

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Combines ranked lists from dense and sparse retrievers using Reciprocal Rank Fusion (RRF).
    
    Formula:
        Score(d) = sum_{system} 1 / (k + rank)
    """
    fused_scores: Dict[int, float] = {}
    doc_registry: Dict[int, Dict[str, Any]] = {}
    dense_ranks: Dict[int, int] = {}
    sparse_ranks: Dict[int, int] = {}

    # Dense scoring
    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        doc_registry[cid] = item
        dense_ranks[cid] = rank
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sparse scoring
    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        if cid not in doc_registry:
            doc_registry[cid] = item
        sparse_ranks[cid] = rank
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sort candidates by combined RRF score descending
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for cid, rrf_score in sorted_items[:top_k]:
        base_item = doc_registry[cid].copy()
        base_item["rrf_score"] = rrf_score
        base_item["dense_rank"] = dense_ranks.get(cid, None)
        base_item["sparse_rank"] = sparse_ranks.get(cid, None)
        base_item["retrieval_type"] = "hybrid_rrf"
        final_results.append(base_item)

    return final_results

