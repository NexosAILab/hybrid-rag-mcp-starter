"""
Configuration module for Hybrid RAG MCP Server.
"""
from datetime import date
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedder: Literal["local", "hashing"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rrf_k: int = 60
    as_of: date = date(2026, 11, 1)
    gate_score_threshold: float = 0.30
    gate_coverage_threshold: float = 0.40
    gate_margin_ratio: float = 0.05
    corpus_dir: Path = Path("data/corpus")
    top_k_retrieval: int = 10
    top_k_answer: int = 5
