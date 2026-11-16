"""
run_benchmark.py — Reproducible evaluation harness comparing Naive, Hybrid, and Hybrid+Gate.
"""
import os
import sys
import json
import argparse
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.corpus import Doc, load_corpus
from hybrid_rag_mcp.filters import apply_temporal_filter
from hybrid_rag_mcp.bm25 import BM25Index
from hybrid_rag_mcp.embeddings import HashingEmbedder, LocalSTEmbedder
from hybrid_rag_mcp.vector_index import VectorIndex
from hybrid_rag_mcp.fusion import reciprocal_rank_fusion, FusedResult
from hybrid_rag_mcp.answer import compose_answer
from hybrid_rag_mcp.gate import evaluate_gate

def run_evaluation(embedder_type: str = "local"):
    settings = Settings(embedder=embedder_type)
    
    corpus_path = Path("data/corpus")
    if not corpus_path.exists():
        corpus_path = Path(__file__).parent.parent / "data/corpus"
    eval_file = Path("data/queries_eval.json")
    if not eval_file.exists():
        eval_file = Path(__file__).parent.parent / "data/queries_eval.json"

    all_docs = load_corpus(corpus_path)
    all_docs_by_id = {d.id: d for d in all_docs}
    
    # Filtered corpus at as_of
    filtered_docs = apply_temporal_filter(all_docs, settings.as_of)
    filtered_docs_by_id = {d.id: d for d in filtered_docs}

    # Embedder setup
    if embedder_type == "hashing":
        embedder = HashingEmbedder()
    else:
        embedder = LocalSTEmbedder(settings.embedding_model)

    # 1. Naive index (vector search ONLY, NO temporal filter, over ALL docs)
    vec_naive = VectorIndex(all_docs, embedder)

    # 2. Filtered indexes (BM25 + Vector over filtered_docs)
    bm25_filtered = BM25Index(filtered_docs)
    vec_filtered = VectorIndex(filtered_docs, embedder)

    with open(eval_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    configs = ["naive", "hybrid", "hybrid_gate"]
    categories = ["exact_id", "temporal", "paraphrase", "unanswerable"]

    scores = {cfg: {cat: 0 for cat in categories} for cfg in configs}
    cat_counts = {cat: 0 for cat in categories}

    detailed_results = []

    for q in queries:
        qid = q["id"]
        cat = q["category"]
        query_text = q["query"]
        expected_doc = q.get("expected_doc")
        expected_abstain = q.get("expected_abstain", False)
        must_contain = q.get("must_contain")
        cat_counts[cat] += 1

        query_record = {
            "id": qid,
            "category": cat,
            "query": query_text,
            "expected_doc": expected_doc,
            "expected_abstain": expected_abstain,
            "must_contain": must_contain,
            "runs": {}
        }

        # --- CONFIG 1: Naive (Vector only, no temporal filter, no gate) ---
        naive_hits = vec_naive.search(query_text, top_k=1)
        top1_naive_id = naive_hits[0][0] if naive_hits else None
        
        # Extractive answer over top1
        fused_naive = [FusedResult(top1_naive_id, 1.0, 1.0, {"vector": 1})] if top1_naive_id else []
        ans_naive = compose_answer(query_text, fused_naive, all_docs_by_id)
        
        naive_correct = False
        if not expected_abstain and top1_naive_id == expected_doc:
            if must_contain:
                naive_correct = must_contain.lower() in ans_naive.text.lower()
            else:
                naive_correct = True
        if naive_correct:
            scores["naive"][cat] += 1
            
        query_record["runs"]["naive"] = {
            "top_doc": top1_naive_id,
            "correct": naive_correct,
            "answer": ans_naive.text
        }

        # --- CONFIG 2: Hybrid (BM25 + Vector + RRF + Temporal filter, no gate) ---
        bm_res = bm25_filtered.search(query_text, top_k=settings.top_k_retrieval)
        vc_res = vec_filtered.search(query_text, top_k=settings.top_k_retrieval)
        fused_hybrid = reciprocal_rank_fusion(
            [bm_res, vc_res],
            k=settings.rrf_k,
            list_names=["bm25", "vector"],
            weights=[1.25, 1.0]
        )
        top1_hybrid_id = fused_hybrid[0].doc_id if fused_hybrid else None
        ans_hybrid = compose_answer(query_text, fused_hybrid, filtered_docs_by_id)

        hybrid_correct = False
        if not expected_abstain and top1_hybrid_id == expected_doc:
            if must_contain:
                hybrid_correct = must_contain.lower() in ans_hybrid.text.lower()
            else:
                hybrid_correct = True
        if hybrid_correct:
            scores["hybrid"][cat] += 1

        query_record["runs"]["hybrid"] = {
            "top_doc": top1_hybrid_id,
            "correct": hybrid_correct,
            "answer": ans_hybrid.text
        }

        # --- CONFIG 3: Hybrid + Gate (Full pipeline) ---
        gate_res = evaluate_gate(
            query=query_text,
            fused=fused_hybrid,
            docs_by_id=filtered_docs_by_id,
            bm25_index=bm25_filtered,
            answer_claims=ans_hybrid.claims,
            gate_score_threshold=settings.gate_score_threshold,
            gate_coverage_threshold=settings.gate_coverage_threshold,
            gate_margin_ratio=settings.gate_margin_ratio
        )

        hybrid_gate_correct = False
        if expected_abstain:
            if not gate_res.passed:
                hybrid_gate_correct = True
        else:
            if gate_res.passed and top1_hybrid_id == expected_doc:
                if must_contain:
                    hybrid_gate_correct = must_contain.lower() in ans_hybrid.text.lower()
                else:
                    hybrid_gate_correct = True
        if hybrid_gate_correct:
            scores["hybrid_gate"][cat] += 1

        query_record["runs"]["hybrid_gate"] = {
            "top_doc": top1_hybrid_id,
            "gate_passed": gate_res.passed,
            "gate_reason": gate_res.reason,
            "correct": hybrid_gate_correct,
            "answer": ans_hybrid.text if gate_res.passed else "ABSTAINED"
        }

        detailed_results.append(query_record)

    # Compute overall
    total_queries = sum(cat_counts.values())
    overall_scores = {
        cfg: sum(scores[cfg][cat] for cat in categories)
        for cfg in configs
    }

    # Generate Markdown Table
    table_lines = [
        "| Configuration | exact_id (6) | temporal (6) | paraphrase (6) | unanswerable (6) | overall (24) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for cfg in configs:
        row = f"| **{cfg}** | {scores[cfg]['exact_id']}/6 | {scores[cfg]['temporal']}/6 | {scores[cfg]['paraphrase']}/6 | {scores[cfg]['unanswerable']}/6 | **{overall_scores[cfg]}/24** |"
        table_lines.append(row)

    md_table = "\n".join(table_lines)

    # Generate full RESULTS.md content
    results_md = f"""# Benchmark Results: Hybrid RAG vs Naive Vector Search

Evaluation executed over the seeded corpus with reproducible configuration.

## Environment & Hyperparameters

- **Embedder:** {embedder_type}
- **Embedding Model:** {settings.embedding_model if embedder_type == "local" else "Deterministic Hashing (384-dim)"}
- **As Of Date:** {settings.as_of.isoformat()}
- **Corpus Size:** {len(all_docs)} total ({len(filtered_docs)} active as of {settings.as_of.isoformat()})
- **RRF Constant (k):** {settings.rrf_k}
- **Gate Thresholds:**
  - `score_norm >= {settings.gate_score_threshold}`
  - `coverage >= {settings.gate_coverage_threshold}`
  - `margin_ratio >= {settings.gate_margin_ratio}`

## Accuracy Comparison

{md_table}

## Per-Query Execution Log

| ID | Category | Expected Doc | Naive Top Doc | Hybrid Top Doc | Gate Passed? | Gate Reason | Final Correct? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in detailed_results:
        q_id = r["id"]
        cat = r["category"]
        exp = r["expected_doc"] or "ABSTAIN"
        n_top = r["runs"]["naive"]["top_doc"] or "-"
        h_top = r["runs"]["hybrid"]["top_doc"] or "-"
        g_pass = "Yes" if r["runs"]["hybrid_gate"]["gate_passed"] else "No"
        g_reason = r["runs"]["hybrid_gate"]["gate_reason"]
        g_corr = "PASS" if r["runs"]["hybrid_gate"]["correct"] else "FAIL"
        results_md += f"| {q_id} | {cat} | {exp} | {n_top} | {h_top} | {g_pass} | {g_reason} | {g_corr} |\n"

    # Write RESULTS.md
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "RESULTS.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(results_md)

    print("\n" + md_table + "\n")
    print(f"[+] Benchmark results written to {out_path}")
    return results_md

def main():
    parser = argparse.ArgumentParser(description="Run reproducible Hybrid RAG benchmark")
    parser.add_argument("--embedder", choices=["local", "hashing"], default="local",
                        help="Embedder to evaluate: 'local' (sentence-transformers) or 'hashing'")
    args = parser.parse_args()
    run_evaluation(args.embedder)

if __name__ == "__main__":
    main()
