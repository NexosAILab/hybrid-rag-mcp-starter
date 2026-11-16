"""
Document schema and corpus loader.
"""
import json
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel

class Doc(BaseModel):
    id: str
    title: str
    service: str
    type: Literal["runbook", "policy", "reference", "filler"]
    body: str
    entities: List[str] = []
    valid_from: date
    superseded_by: Optional[str] = None

def load_corpus(corpus_dir: Path) -> List[Doc]:
    docs: List[Doc] = []
    path = Path(corpus_dir)
    if not path.exists():
        return docs
    for json_file in path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs.append(Doc.model_validate(data))
    docs.sort(key=lambda d: d.id)
    return docs
