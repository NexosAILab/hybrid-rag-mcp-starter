"""
Temporal validity filters for corpus documents.
"""
from datetime import date
from typing import Dict, List
from .corpus import Doc

def apply_temporal_filter(docs: List[Doc], as_of: date) -> List[Doc]:
    docs_by_id: Dict[str, Doc] = {d.id: d for d in docs}
    valid_docs: List[Doc] = []

    for d in docs:
        if d.valid_from > as_of:
            continue
        if d.superseded_by is None:
            valid_docs.append(d)
        else:
            successor = docs_by_id.get(d.superseded_by)
            if successor is None or successor.valid_from > as_of:
                valid_docs.append(d)

    return sorted(valid_docs, key=lambda x: x.id)
