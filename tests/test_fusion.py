import pytest
from hybrid_rag_mcp.fusion import reciprocal_rank_fusion

def test_rrf_scoring_and_ordering():
    list1 = [("DOC-A", 10.0), ("DOC-B", 5.0), ("DOC-C", 1.0)]
    list2 = [("DOC-B", 8.0), ("DOC-A", 4.0), ("DOC-D", 2.0)]
    
    # k = 60
    # DOC-A: 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # DOC-B: 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522 (tie breaks by doc_id asc -> DOC-A)
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert fused[0].doc_id == "DOC-A"
    assert fused[1].doc_id == "DOC-B"

def test_rrf_top1_normalization():
    list1 = [("DOC-1", 10.0)]
    list2 = [("DOC-1", 8.0)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert pytest.approx(fused[0].score_norm, 1e-4) == 1.0

def test_rrf_k_monotonicity():
    list1 = [("DOC-1", 10.0), ("DOC-2", 5.0)]
    list2 = [("DOC-1", 8.0), ("DOC-2", 4.0)]
    
    fused_k10 = reciprocal_rank_fusion([list1, list2], k=10)
    fused_k60 = reciprocal_rank_fusion([list1, list2], k=60)
    
    diff_k10 = fused_k10[0].rrf_score - fused_k10[1].rrf_score
    diff_k60 = fused_k60[0].rrf_score - fused_k60[1].rrf_score
    assert diff_k10 > diff_k60
