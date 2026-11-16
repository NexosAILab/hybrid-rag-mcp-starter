"""
Deterministic extractive answer composer.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np

from .bm25 import tokenize
from .corpus import Doc
from .fusion import FusedResult
from .gate import Claim, extract_identifiers, NUMERIC_VALUE_REGEX

@dataclass
class Answer:
    text: str
    claims: List[Claim]

def split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def compose_answer(
    query: str,
    fused: List[FusedResult],
    docs_by_id: Dict[str, Doc]
) -> Answer:
    if not fused:
        return Answer(text="", claims=[])

    top1 = fused[0]
    top_doc = docs_by_id.get(top1.doc_id)
    if not top_doc:
        return Answer(text="", claims=[])

    sentences = split_into_sentences(top_doc.body)
    if not sentences:
        return Answer(text="", claims=[])

    query_tokens = tokenize(query)
    q_ids = extract_identifiers(query)

    scored_sentences = []
    for idx, s in enumerate(sentences):
        s_tokens = set(tokenize(s))
        s_ids = extract_identifiers(s)
        
        id_matches = len(q_ids.intersection(s_ids))
        content_matches = len(set(query_tokens).intersection(s_tokens))
        
        score = 3 * id_matches + content_matches
        scored_sentences.append((score, idx, s))

    # Pick top 2 sentences (ties -> earliest in doc)
    scored_sentences.sort(key=lambda item: (-item[0], item[1]))
    
    # Check if any sentence has content token matches
    best_score = scored_sentences[0][0]
    if best_score <= 0:
        return Answer(text="", claims=[])

    top_candidates = scored_sentences[:2]
    # Preserve original document order
    top_candidates.sort(key=lambda item: item[1])

    chosen_sentences = [item[2] for item in top_candidates]
    composed_text = " ".join(chosen_sentences) + f" [Source: {top_doc.id}]"

    claims: List[Claim] = []
    for s in chosen_sentences:
        claim_tokens: Set[str] = set()
        for ident in extract_identifiers(s):
            claim_tokens.add(ident)
        for num in NUMERIC_VALUE_REGEX.findall(s):
            claim_tokens.add(num.lower())
        claims.append(Claim(text=s, doc_id=top_doc.id, tokens=claim_tokens))

    return Answer(text=composed_text, claims=claims)
