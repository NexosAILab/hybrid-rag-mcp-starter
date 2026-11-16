import os
import pytest
from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.server import init_server, search_knowledge, answer_with_citations, evaluate

@pytest.fixture(autouse=True)
def setup_hashing_server(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "hashing")
    settings = Settings(embedder="hashing")
    init_server(settings)

def test_search_knowledge_hero_query():
    res = search_knowledge("What is the remediation for error ERR_CACHE_109 on the billing service?")
    assert len(res["results"]) > 0
    # BM25 ranks RB-114 top and it appears in search results
    doc_ids = [r["doc_id"] for r in res["results"]]
    assert "RB-114" in doc_ids

def test_answer_unanswerable_query():
    res = answer_with_citations("What is the SLA penalty clause for the EU region?")
    assert res["status"] == "abstained"
    assert res["gate"]["reason"] in ["LOW_CONFIDENCE", "NO_CONTEXT", "ENTITY_NOT_FOUND"]

def test_evaluate_returns_structure():
    eval_res = evaluate()
    assert "overall_accuracy" in eval_res
    assert "categories" in eval_res
    assert "exact_id" in eval_res["categories"]
    assert eval_res["total_queries"] == 24
