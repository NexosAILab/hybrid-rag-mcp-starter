"""
In-house Okapi BM25 implementation.
"""
import math
import re
from typing import Dict, List, Set, Tuple
from .corpus import Doc

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())

class BM25Index:
    def __init__(self, docs: List[Doc], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_ids = [d.id for d in docs]
        self.N = len(docs)
        
        self.doc_len: Dict[str, int] = {}
        self.doc_tfs: Dict[str, Dict[str, int]] = {}
        self._doc_token_sets: Dict[str, Set[str]] = {}
        self.df: Dict[str, int] = {}
        
        total_len = 0
        for doc in docs:
            tokens = tokenize(doc.title + " " + doc.body)
            self.doc_len[doc.id] = len(tokens)
            total_len += len(tokens)
            
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tfs[doc.id] = tf
            
            token_set = set(tokens)
            self._doc_token_sets[doc.id] = token_set
            for t in token_set:
                self.df[t] = self.df.get(t, 0) + 1
                
        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0

    def idf(self, token: str) -> float:
        n_t = self.df.get(token, 0)
        return math.log(1.0 + (self.N - n_t + 0.5) / (n_t + 0.5))

    def doc_tokens(self, doc_id: str) -> Set[str]:
        return self._doc_token_sets.get(doc_id, set())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query)
        unique_q = set(q_tokens)
        
        scores: List[Tuple[str, float]] = []
        for doc_id in self.doc_ids:
            score = 0.0
            d_len = self.doc_len[doc_id]
            tf_map = self.doc_tfs[doc_id]
            
            for t in unique_q:
                f_td = tf_map.get(t, 0)
                if f_td > 0:
                    idf_t = self.idf(t)
                    denom = f_td + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl)) if self.avgdl > 0 else 1.0
                    num = f_td * (self.k1 + 1.0)
                    score += idf_t * (num / denom)
            scores.append((doc_id, score))
            
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]
