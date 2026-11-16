from datetime import date
from pathlib import Path
from hybrid_rag_mcp.corpus import Doc, load_corpus
from hybrid_rag_mcp.filters import apply_temporal_filter

def test_temporal_filter_as_of_nov_2026():
    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists():
        corpus_dir = Path(__file__).parent.parent / "data/corpus"
    docs = load_corpus(corpus_dir)
    
    filtered = apply_temporal_filter(docs, date(2026, 11, 1))
    filtered_ids = set(d.id for d in filtered)
    
    assert "POL-2024-07" not in filtered_ids
    assert "POL-2026-02" in filtered_ids

def test_temporal_filter_as_of_mid_2025():
    corpus_dir = Path("data/corpus")
    if not corpus_dir.exists():
        corpus_dir = Path(__file__).parent.parent / "data/corpus"
    docs = load_corpus(corpus_dir)
    
    filtered = apply_temporal_filter(docs, date(2025, 6, 1))
    filtered_ids = set(d.id for d in filtered)
    
    # POL-2024-07 is valid from 2024-06-01 and successor POL-2026-02 valid from 2026-02-01 > 2025-06-01
    assert "POL-2024-07" in filtered_ids
    assert "POL-2026-02" not in filtered_ids

def test_future_doc_excluded():
    doc = Doc(id="DOC-FUTURE", title="Future spec", service="general", type="runbook",
              body="Future instructions.", valid_from="2030-01-01")
    filtered = apply_temporal_filter([doc], date(2026, 11, 1))
    assert len(filtered) == 0
