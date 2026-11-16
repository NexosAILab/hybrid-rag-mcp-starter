import pytest
from hybrid_rag_mcp.bm25 import tokenize, BM25Index
from hybrid_rag_mcp.corpus import Doc

def test_tokenizer_keeps_identifiers_intact():
    text = "Error ERR_CACHE_109 occurred on SKU-8841-KX variant."
    tokens = tokenize(text)
    assert "err_cache_109" in tokens
    assert "sku-8841-kx" in tokens

def test_bm25_exact_id_ranking_fixture():
    docs = [
        Doc(id="DOC-CORRECT", title="Cache error remediation", service="billing", type="runbook",
            body="For ERR_CACHE_109, purge session cache and rotate key K-7.", valid_from="2025-01-01"),
        Doc(id="DOC-DISTRACTOR", title="Cache error remediation steps and recovery", service="billing", type="runbook",
            body="For ERR_CACHE_190, restart cache cluster and flush all nodes.", valid_from="2025-01-01"),
        Doc(id="DOC-UNRELATED", title="Database backup runbook", service="platform", type="reference",
            body="Nightly snapshots and point-in-time recovery restore corrupt data.", valid_from="2025-01-01"),
    ]
    index = BM25Index(docs)
    results = index.search("What is the fix for ERR_CACHE_109?", top_k=3)
    assert results[0][0] == "DOC-CORRECT"

def test_bm25_tie_breaking():
    docs = [
        Doc(id="DOC-B", title="Unrelated doc", service="general", type="filler",
            body="No matching keywords here.", valid_from="2025-01-01"),
        Doc(id="DOC-A", title="Unrelated doc", service="general", type="filler",
            body="No matching keywords here.", valid_from="2025-01-01"),
    ]
    index = BM25Index(docs)
    results = index.search("query with zero match", top_k=2)
    assert results[0][0] == "DOC-A"
    assert results[1][0] == "DOC-B"
