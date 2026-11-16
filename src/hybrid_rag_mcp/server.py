"""
FastMCP Server implementation exposing search, answer_with_citations, and evaluate.
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

from .config import Settings
from .corpus import Doc, load_corpus
from .filters import apply_temporal_filter
from .bm25 import BM25Index
from .embeddings import HashingEmbedder, LocalSTEmbedder
from .vector_index import VectorIndex
from .fusion import reciprocal_rank_fusion
from .answer import compose_answer
from .gate import evaluate_gate

mcp = FastMCP("hybrid-rag-mcp")

# Module state
_settings: Optional[Settings] = None
_docs_by_id: Dict[str, Doc] = {}
_filtered_docs: List[Doc] = []
_bm25_index: Optional[BM25Index] = None
_vector_index: Optional[VectorIndex] = None

def init_server(settings: Optional[Settings] = None):
    global _settings, _docs_by_id, _filtered_docs, _bm25_index, _vector_index
    _settings = settings or Settings()
    
    # Load corpus
    corpus_path = _settings.corpus_dir
    if not corpus_path.is_absolute():
        corpus_path = Path.cwd() / corpus_path
        if not corpus_path.exists():
            pkg_root = Path(__file__).parent.parent.parent
            corpus_path = pkg_root / _settings.corpus_dir
            
    all_docs = load_corpus(corpus_path)
    _docs_by_id = {d.id: d for d in all_docs}
    _filtered_docs = apply_temporal_filter(all_docs, _settings.as_of)
    
    # Build BM25
    _bm25_index = BM25Index(_filtered_docs)
    
    # Build Vector Index
    if _settings.embedder == "hashing":
        embedder = HashingEmbedder()
    else:
        embedder = LocalSTEmbedder(_settings.embedding_model)
    _vector_index = VectorIndex(_filtered_docs, embedder)

def get_state():
    if _bm25_index is None:
        init_server()
    return _settings, _docs_by_id, _filtered_docs, _bm25_index, _vector_index

def search_knowledge(query: str, top_k: int = 5) -> Dict[str, Any]:
    settings, docs_by_id, filtered_docs, bm25_idx, vec_idx = get_state()
    
    bm25_results = bm25_idx.search(query, top_k=settings.top_k_retrieval)
    vec_results = vec_idx.search(query, top_k=settings.top_k_retrieval)
    
    fused = reciprocal_rank_fusion(
        [bm25_results, vec_results],
        k=settings.rrf_k,
        list_names=["bm25", "vector"]
    )
    
    output_results = []
    for fr in fused[:top_k]:
        doc = docs_by_id.get(fr.doc_id)
        title = doc.title if doc else ""
        output_results.append({
            "doc_id": fr.doc_id,
            "title": title,
            "score_norm": round(fr.score_norm, 4),
            "bm25_rank": fr.ranks.get("bm25"),
            "vector_rank": fr.ranks.get("vector")
        })
        
    return {"results": output_results}

def answer_with_citations(question: str) -> Dict[str, Any]:
    settings, docs_by_id, filtered_docs, bm25_idx, vec_idx = get_state()
    
    bm25_results = bm25_idx.search(question, top_k=settings.top_k_retrieval)
    vec_results = vec_idx.search(question, top_k=settings.top_k_retrieval)
    
    fused = reciprocal_rank_fusion(
        [bm25_results, vec_results],
        k=settings.rrf_k,
        list_names=["bm25", "vector"]
    )
    
    ans = compose_answer(question, fused, docs_by_id)
    
    gate_res = evaluate_gate(
        query=question,
        fused=fused,
        docs_by_id=docs_by_id,
        bm25_index=bm25_idx,
        answer_claims=ans.claims,
        gate_score_threshold=settings.gate_score_threshold,
        gate_coverage_threshold=settings.gate_coverage_threshold,
        gate_margin_ratio=settings.gate_margin_ratio
    )
    
    if gate_res.passed:
        top_doc = fused[0].doc_id if fused else None
        citations = [top_doc] if top_doc else []
        return {
            "status": "answered",
            "answer": ans.text,
            "citations": citations,
            "gate": {
                "score_norm": round(gate_res.details.get("score_norm", 0.0), 4),
                "coverage": round(gate_res.details.get("coverage", 0.0), 4),
                "reason": gate_res.reason
            }
        }
    else:
        return {
            "status": "abstained",
            "answer": "Insufficient grounded context to answer this question from the knowledge base.",
            "citations": [],
            "gate": {
                "score_norm": round(gate_res.details.get("score_norm", 0.0), 4),
                "coverage": round(gate_res.details.get("coverage", 0.0), 4),
                "reason": gate_res.reason
            }
        }

def evaluate() -> Dict[str, Any]:
    settings, docs_by_id, _, _, _ = get_state()
    
    eval_file = Path("data/queries_eval.json")
    if not eval_file.exists():
        eval_file = Path(__file__).parent.parent.parent / "data/queries_eval.json"
        
    with open(eval_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    categories = ["exact_id", "temporal", "paraphrase", "unanswerable"]
    cat_correct = {c: 0 for c in categories}
    cat_total = {c: 0 for c in categories}
    safe_abstentions = 0

    for q in queries:
        cat = q["category"]
        cat_total[cat] += 1
        res = answer_with_citations(q["query"])
        
        expected_abstain = q.get("expected_abstain", False)
        expected_doc = q.get("expected_doc")
        must_contain = q.get("must_contain")
        
        if expected_abstain:
            if res["status"] == "abstained":
                cat_correct[cat] += 1
        else:
            if res["status"] == "answered":
                cited_ok = bool(res["citations"] and res["citations"][0] == expected_doc)
                content_ok = True
                if must_contain:
                    content_ok = must_contain.lower() in (res["answer"] or "").lower()
                if cited_ok and content_ok:
                    cat_correct[cat] += 1
            else:
                safe_abstentions += 1

    total_queries = len(queries)
    total_correct = sum(cat_correct.values())
    accuracy = (total_correct / total_queries) if total_queries > 0 else 0.0

    return {
        "overall_accuracy": round(accuracy, 4),
        "total_correct": total_correct,
        "total_queries": total_queries,
        "safe_abstentions": safe_abstentions,
        "categories": {
            c: {
                "correct": cat_correct[c],
                "total": cat_total[c],
                "accuracy": round(cat_correct[c] / cat_total[c], 4) if cat_total[c] > 0 else 0.0
            }
            for c in categories
        }
    }

@mcp.tool()
def search_knowledge_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search knowledge base with hybrid retrieval (BM25 + vectors)."""
    return search_knowledge(query, top_k)

@mcp.tool()
def answer_with_citations_tool(question: str) -> Dict[str, Any]:
    """Generate grounded answer with exact source citations or abstain safely."""
    return answer_with_citations(question)

@mcp.tool()
def evaluate_tool() -> Dict[str, Any]:
    """Run full benchmark evaluation across all labeled queries."""
    return evaluate()

def main():
    init_server()
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
