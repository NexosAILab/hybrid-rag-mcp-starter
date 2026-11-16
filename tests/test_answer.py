from datetime import date
from pathlib import Path
from hybrid_rag_mcp.corpus import load_corpus
from hybrid_rag_mcp.filters import apply_temporal_filter
from hybrid_rag_mcp.fusion import FusedResult
from hybrid_rag_mcp.answer import compose_answer

def test_answer_composition_on_real_corpus():
    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists():
        corpus_dir = Path(__file__).parent.parent / "data/corpus"
    docs = load_corpus(corpus_dir)
    filtered = apply_temporal_filter(docs, date(2026, 11, 1))
    docs_by_id = {d.id: d for d in filtered}

    # (a) Refund query
    fused_refund = [FusedResult("POL-2026-02", 0.03, 1.0, {})]
    ans_refund = compose_answer("What is the refund window for annual plans?", fused_refund, docs_by_id)
    assert "14 days" in ans_refund.text
    assert "[Source: POL-2026-02]" in ans_refund.text

    # (b) ERR_CACHE_109 query
    fused_cache = [FusedResult("RB-114", 0.03, 1.0, {})]
    ans_cache = compose_answer("What is the remediation for error ERR_CACHE_109 on the billing service?",
                               fused_cache, docs_by_id)
    assert "rotate signing key K-7" in ans_cache.text
    assert "[Source: RB-114]" in ans_cache.text

    # (c) Every claim's doc_id is in fused list
    fused_ids = set(fr.doc_id for fr in fused_cache)
    for c in ans_cache.claims:
        assert c.doc_id in fused_ids
