"""
Embeddings interfaces and implementations.
"""
import hashlib
from typing import List, Protocol
import numpy as np
from .bm25 import tokenize

class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...

class HashingEmbedder:
    """Zero-dependency deterministic hashing embedder."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        mat = np.zeros((n, self.dim), dtype=np.float32)
        
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokens:
                h = hashlib.sha256(token.encode("utf-8")).hexdigest()
                idx = int(h[:4], 16) % self.dim
                sign = 1.0 if (int(h[4], 16) % 2 == 1) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            mat[i] = vec
            
        return mat

class LocalSTEmbedder:
    """Local embedder wrapping sentence-transformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32)
