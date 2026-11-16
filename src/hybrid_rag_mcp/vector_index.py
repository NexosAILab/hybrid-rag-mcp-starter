"""
Vector Index implementation using dot-product cosine similarity.
"""
from typing import List, Tuple
import numpy as np
from .corpus import Doc
from .embeddings import Embedder

class VectorIndex:
    def __init__(self, docs: List[Doc], embedder: Embedder):
        self.docs = docs
        self.doc_ids = [d.id for d in docs]
        self.embedder = embedder
        
        texts = [d.title + " " + d.body for d in docs]
        if texts:
            self.matrix = self.embedder.embed(texts)
        else:
            self.matrix = np.zeros((0, 384), dtype=np.float32)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if len(self.doc_ids) == 0:
            return []
            
        q_vec = self.embedder.embed([query])[0]
        scores = np.dot(self.matrix, q_vec)
        
        results: List[Tuple[str, float]] = []
        for doc_id, score in zip(self.doc_ids, scores):
            results.append((doc_id, float(score)))
            
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]
