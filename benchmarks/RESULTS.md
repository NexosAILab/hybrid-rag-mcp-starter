# Benchmark Results: Hybrid RAG vs Naive Vector Search

Evaluation executed over the seeded corpus with reproducible configuration.

## Environment & Hyperparameters

- **Embedder:** local
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **As Of Date:** 2026-11-01
- **Corpus Size:** 48 total (45 active as of 2026-11-01)
- **RRF Constant (k):** 60
- **Gate Thresholds:**
  - `score_norm >= 0.3`
  - `coverage >= 0.5`
  - `margin_ratio >= 0.1`

## Accuracy Comparison

| Configuration | exact_id (6) | temporal (6) | paraphrase (6) | unanswerable (6) | overall (24) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **naive** | 4/6 | 2/6 | 6/6 | 0/6 | **12/24** |
| **hybrid** | 5/6 | 4/6 | 6/6 | 0/6 | **15/24** |
| **hybrid_gate** | 0/6 | 2/6 | 4/6 | 6/6 | **12/24** |

## Per-Query Execution Log

| ID | Category | Expected Doc | Naive Top Doc | Hybrid Top Doc | Gate Passed? | Gate Reason | Final Correct? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Q01 | exact_id | RB-114 | RB-109 | RB-109 | No | ENTITY_NOT_FOUND | FAIL |
| Q02 | exact_id | RB-207 | RB-208 | RB-207 | No | LOW_CONFIDENCE | FAIL |
| Q03 | exact_id | RB-311 | RB-311 | RB-311 | No | AMBIGUOUS_MATCH | FAIL |
| Q04 | exact_id | RB-402 | RB-402 | RB-402 | No | LOW_CONFIDENCE | FAIL |
| Q05 | exact_id | RB-518 | RB-518 | RB-518 | No | AMBIGUOUS_MATCH | FAIL |
| Q06 | exact_id | RB-620 | RB-620 | RB-620 | No | AMBIGUOUS_MATCH | FAIL |
| Q07 | temporal | POL-2026-02 | POL-2026-02 | POL-2026-02 | Yes | OK | PASS |
| Q08 | temporal | POL-2026-02 | POL-2026-02 | POL-2026-02 | No | LOW_CONFIDENCE | FAIL |
| Q09 | temporal | POL-2026-01 | POL-2024-11 | POL-2026-01 | Yes | OK | PASS |
| Q10 | temporal | POL-2026-01 | POL-2024-11 | POL-2026-01 | No | LOW_CONFIDENCE | FAIL |
| Q11 | temporal | POL-2026-03 | POL-2026-03 | POL-2026-03 | Yes | OK | FAIL |
| Q12 | temporal | POL-2026-03 | POL-2026-03 | POL-2026-03 | Yes | OK | FAIL |
| Q13 | paraphrase | REF-001 | REF-001 | REF-001 | Yes | OK | PASS |
| Q14 | paraphrase | REF-002 | REF-002 | REF-002 | No | LOW_CONFIDENCE | FAIL |
| Q15 | paraphrase | REF-003 | REF-003 | REF-003 | No | LOW_CONFIDENCE | FAIL |
| Q16 | paraphrase | REF-004 | REF-004 | REF-004 | Yes | OK | PASS |
| Q17 | paraphrase | REF-005 | REF-005 | REF-005 | Yes | OK | PASS |
| Q18 | paraphrase | REF-006 | REF-006 | REF-006 | Yes | OK | PASS |
| Q19 | unanswerable | ABSTAIN | REF-004 | REF-004 | No | LOW_CONFIDENCE | PASS |
| Q20 | unanswerable | ABSTAIN | RB-620 | RB-620 | No | LOW_CONFIDENCE | PASS |
| Q21 | unanswerable | ABSTAIN | F-014 | POL-2026-02 | No | LOW_CONFIDENCE | PASS |
| Q22 | unanswerable | ABSTAIN | F-015 | F-015 | No | LOW_CONFIDENCE | PASS |
| Q23 | unanswerable | ABSTAIN | POL-2026-02 | F-012 | No | LOW_CONFIDENCE | PASS |
| Q24 | unanswerable | ABSTAIN | REF-004 | REF-004 | No | LOW_CONFIDENCE | PASS |
