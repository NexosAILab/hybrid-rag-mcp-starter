"""
Reciprocal Rank Fusion (RRF) module.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class FusedResult:
    doc_id: str
    rrf_score: float
    score_norm: float
    ranks: Dict[str, int]

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    list_names: Optional[List[str]] = None,
    weights: Optional[List[float]] = None
) -> List[FusedResult]:
    if list_names is None:
        list_names = [f"list_{i}" for i in range(len(rankings))]

    if weights is None:
        weights = [1.0] * len(rankings)

    rrf_scores: Dict[str, float] = {}
    doc_ranks: Dict[str, Dict[str, int]] = {}

    for name, r_list, w in zip(list_names, rankings, weights):
        for rank_zero, (doc_id, _) in enumerate(r_list):
            rank = rank_zero + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (w / (k + rank))
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
            doc_ranks[doc_id][name] = rank

    max_possible = sum(w for w in weights) / (k + 1.0) if weights else 1.0

    results: List[FusedResult] = []
    for doc_id, score in rrf_scores.items():
        score_norm = score / max_possible if max_possible > 0 else 0.0
        results.append(
            FusedResult(
                doc_id=doc_id,
                rrf_score=score,
                score_norm=score_norm,
                ranks=doc_ranks.get(doc_id, {})
            )
        )

    results.sort(key=lambda item: (-item.rrf_score, item.doc_id))
    return results
