"""
Evaluation Gate module.
Enforces deterministic grounding checks before emitting answers.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np

from .bm25 import BM25Index, tokenize
from .corpus import Doc
from .fusion import FusedResult

IDENTIFIER_REGEX = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+_\d+|[A-Z]+-\d+-[A-Z0-9]+)\b"
)
SECTION_REGEX = re.compile(r"Section\s+\d+\.\d+", re.IGNORECASE)
NUMERIC_VALUE_REGEX = re.compile(r"\b\d+\s*(?:days|hours|months|minutes)\b", re.IGNORECASE)

def extract_identifiers(text: str) -> Set[str]:
    ids: Set[str] = set()
    for m in IDENTIFIER_REGEX.findall(text):
        ids.add(m.lower())
    for s in SECTION_REGEX.findall(text):
        normalized = re.sub(r"\s+", "", s).lower()
        ids.add(normalized)
    return ids

@dataclass
class Claim:
    text: str
    doc_id: str
    tokens: Set[str]

@dataclass
class GateResult:
    passed: bool
    reason: str
    details: Dict[str, Any]

def evaluate_gate(
    query: str,
    fused: List[FusedResult],
    docs_by_id: Dict[str, Doc],
    bm25_index: BM25Index,
    answer_claims: List[Claim],
    gate_score_threshold: float = 0.30,
    gate_coverage_threshold: float = 0.50,
    gate_margin_ratio: float = 0.10
) -> GateResult:
    # 1. NO_CONTEXT
    if not fused:
        return GateResult(
            passed=False,
            reason="NO_CONTEXT",
            details={"error": "Fused retrieval results list is empty"}
        )

    top1_result = fused[0]
    top1_doc = docs_by_id.get(top1_result.doc_id)
    if not top1_doc:
        return GateResult(
            passed=False,
            reason="NO_CONTEXT",
            details={"error": f"Top document {top1_result.doc_id} not found in corpus"}
        )

    # 2. ENTITY_NOT_FOUND
    query_identifiers = extract_identifiers(query)
    top1_tokens = bm25_index.doc_tokens(top1_doc.id)
    top1_text = (top1_doc.title + " " + top1_doc.body).lower()
    top1_norm_text = re.sub(r"\s+", "", top1_text)

    if query_identifiers:
        for q_id in query_identifiers:
            id_in_tokens = q_id in top1_tokens
            id_in_text = q_id in top1_text or q_id in top1_norm_text
            if not (id_in_tokens or id_in_text):
                return GateResult(
                    passed=False,
                    reason="ENTITY_NOT_FOUND",
                    details={"missing_identifier": q_id, "top1_doc": top1_doc.id}
                )

    # 3. LOW_CONFIDENCE (coverage)
    query_tokens = tokenize(query)
    if query_tokens:
        idfs = [bm25_index.idf(t) for t in query_tokens]
        median_idf = float(np.median(idfs)) if idfs else 0.0
        
        content_tokens = set(
            t for t, score in zip(query_tokens, idfs) if score >= median_idf
        )
        for q_id in query_identifiers:
            content_tokens.update(tokenize(q_id))

        total_idf_sum = sum(bm25_index.idf(t) for t in content_tokens)
        matched_tokens = content_tokens.intersection(top1_tokens)
        matched_idf_sum = sum(bm25_index.idf(t) for t in matched_tokens)
        
        coverage = (matched_idf_sum / total_idf_sum) if total_idf_sum > 0 else 0.0
    else:
        coverage = 0.0

    if coverage < gate_coverage_threshold:
        return GateResult(
            passed=False,
            reason="LOW_CONFIDENCE",
            details={"type": "coverage", "coverage": coverage, "threshold": gate_coverage_threshold}
        )

    # 4. LOW_CONFIDENCE (score)
    if top1_result.score_norm < gate_score_threshold:
        return GateResult(
            passed=False,
            reason="LOW_CONFIDENCE",
            details={"type": "score", "score_norm": top1_result.score_norm, "threshold": gate_score_threshold}
        )

    # 5. AMBIGUOUS_MATCH
    if query_identifiers and len(fused) > 1:
        runner_up = fused[1]
        runner_doc = docs_by_id.get(runner_up.doc_id)
        if runner_doc and top1_result.score_norm > 0:
            margin = (top1_result.score_norm - runner_up.score_norm) / top1_result.score_norm
            if margin < gate_margin_ratio:
                runner_ids = extract_identifiers(runner_doc.title + " " + runner_doc.body)
                if runner_ids and (runner_ids != query_identifiers):
                    return GateResult(
                        passed=False,
                        reason="AMBIGUOUS_MATCH",
                        details={"margin": margin, "top1": top1_doc.id, "runner_up": runner_doc.id}
                    )

    # 6. CITATION_MISMATCH
    fused_doc_ids = set(fr.doc_id for fr in fused)
    for claim in answer_claims:
        if claim.doc_id not in fused_doc_ids:
            return GateResult(
                passed=False,
                reason="CITATION_MISMATCH",
                details={"error": f"Claim doc_id {claim.doc_id} not present in fused results"}
            )
        cited_doc = docs_by_id.get(claim.doc_id)
        if not cited_doc:
            return GateResult(
                passed=False,
                reason="CITATION_MISMATCH",
                details={"error": f"Cited doc {claim.doc_id} not found in docs map"}
            )
        
        cited_body_lower = cited_doc.body.lower()
        cited_norm_body = re.sub(r"\s+", "", cited_body_lower)

        for token in claim.tokens:
            tok_lower = token.lower()
            norm_tok = re.sub(r"\s+", "", tok_lower)
            if (tok_lower not in cited_body_lower) and (norm_tok not in cited_norm_body):
                return GateResult(
                    passed=False,
                    reason="CITATION_MISMATCH",
                    details={"missing_claim_token": token, "cited_doc": claim.doc_id}
                )

    return GateResult(
        passed=True,
        reason="OK",
        details={
            "score_norm": top1_result.score_norm,
            "coverage": coverage,
            "top1_doc": top1_doc.id
        }
    )
