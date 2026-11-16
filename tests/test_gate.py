from hybrid_rag_mcp.bm25 import BM25Index
from hybrid_rag_mcp.corpus import Doc
from hybrid_rag_mcp.fusion import FusedResult
from hybrid_rag_mcp.gate import evaluate_gate, Claim

def test_gate_reasons_coverage():
    docs = [
        Doc(id="RB-114", title="Cache error", service="billing", type="runbook",
            body="For ERR_CACHE_109, purge session cache and rotate signing key K-7.", valid_from="2025-01-01"),
        Doc(id="RB-109", title="Cache error distractor", service="billing", type="runbook",
            body="For ERR_CACHE_190, restart cache cluster.", valid_from="2025-01-01"),
        Doc(id="POL-2026-02", title="Refund Policy", service="billing", type="policy",
            body="The refund window for annual plans is 14 days from the invoice date.", valid_from="2026-02-01"),
    ]
    docs_by_id = {d.id: d for d in docs}
    bm25 = BM25Index(docs)

    # 1. NO_CONTEXT
    res_no_context = evaluate_gate("What is ERR_CACHE_109?", [], docs_by_id, bm25, [])
    assert res_no_context.reason == "NO_CONTEXT"

    # 2. ENTITY_NOT_FOUND (ID query, top doc missing the ID)
    fused_wrong_entity = [FusedResult("RB-109", 0.05, 0.9, {})]
    res_entity = evaluate_gate("What is ERR_CACHE_109?", fused_wrong_entity, docs_by_id, bm25, [])
    assert res_entity.reason == "ENTITY_NOT_FOUND"

    # 3. LOW_CONFIDENCE (coverage)
    doc_generic = Doc(id="DOC-GENERIC", title="General statement", service="general", type="filler",
                      body="Something completely unrelated to this query.", valid_from="2025-01-01")
    d_map = {**docs_by_id, "DOC-GENERIC": doc_generic}
    bm25_gen = BM25Index(list(d_map.values()))
    fused_generic = [FusedResult("DOC-GENERIC", 0.05, 0.9, {})]
    res_cov = evaluate_gate("How do we recover the database after a bad deploy corrupts data?",
                            fused_generic, d_map, bm25_gen, [])
    assert res_cov.reason == "LOW_CONFIDENCE"
    assert res_cov.details.get("type") == "coverage"

    # 4. AMBIGUOUS_MATCH
    # Construct runner-up within margin and with query matching top doc tokens to satisfy coverage
    fused_ambig = [
        FusedResult("RB-114", 0.032, 0.95, {}),
        FusedResult("RB-109", 0.031, 0.93, {})  # margin = (0.95 - 0.93)/0.95 = 0.021 < 0.10
    ]
    # Query with exact tokens present in RB-114 to pass coverage check
    res_ambig = evaluate_gate("purge session cache and rotate signing key ERR_CACHE_109", fused_ambig, docs_by_id, bm25, [])
    assert res_ambig.reason == "AMBIGUOUS_MATCH"

    # 5. CITATION_MISMATCH
    claim_bad = Claim(text="The refund window is 30 days.", doc_id="POL-2026-02", tokens={"30 days"})
    fused_pol = [FusedResult("POL-2026-02", 0.04, 0.95, {})]
    res_cite = evaluate_gate("What is the refund window for annual plans?", fused_pol, docs_by_id, bm25, [claim_bad])
    assert res_cite.reason == "CITATION_MISMATCH"

    # 6. Clean PASS
    claim_good = Claim(text="The refund window for annual plans is 14 days.", doc_id="POL-2026-02", tokens={"14 days"})
    res_pass = evaluate_gate("What is the refund window for annual plans?", fused_pol, docs_by_id, bm25, [claim_good])
    assert res_pass.passed is True
    assert res_pass.reason == "OK"
