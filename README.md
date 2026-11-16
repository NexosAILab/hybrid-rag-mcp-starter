# hybrid-rag-mcp-starter

Production-ready Hybrid Retrieval-Augmented Generation (RAG) MCP Server in pure Python, combining deterministic keyword search (BM25), dense vector embeddings (Reciprocal Rank Fusion, $k=60$), temporal policy filtering, and a strict verification gate before returning grounded answers.

![Nexos AI Lab Banner](https://raw.githubusercontent.com/NexosAILab/.github/main/brand-assets/github_header_banner.jpg)

[![YouTube](https://img.shields.io/badge/YouTube-Nexos%20AI%20Lab-red.svg?logo=youtube)](https://www.youtube.com/@NexosAILab)
[![X](https://img.shields.io/badge/X-@NexosAILab-black.svg?logo=x)](https://x.com/NexosAILab)
[![Web Hub](https://img.shields.io/badge/Web%20Hub-nexosailab.vercel.app-cyan.svg)](https://nexosailab.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview & The 3 Failure Modes of Naive RAG

Most enterprise RAG pipelines fail in production because they rely solely on approximate nearest neighbor vector search:

1. **Exact Identifier Blindness:** Dense embeddings struggle with exact error codes, ticket IDs, or part numbers (e.g. confusing `ERR_CACHE_109` with distractor `ERR_CACHE_108`), returning wrong remediation steps with high confidence.
2. **Temporal Supersedence Trap:** Outdated policies or superseded runbooks with high semantic similarity outrank newer, active documentation.
3. **Hallucination on Missing Context:** When the corpus does not contain the answer, vector search still returns the nearest geometric neighbors, prompting LLMs to invent answers.

`hybrid-rag-mcp-starter` solves all three failure modes deterministically:
- **In-house BM25** guarantees exact token matches for IDs, codes, and acronyms.
- **Temporal Filter Engine** enforces `as_of` date bounds and resolves `supersedes` chains before retrieval.
- **Reciprocal Rank Fusion (RRF, $k=60$)** merges sparse and dense rankings robustly without manual score calibration.
- **Deterministic Gate** evaluates top-rank margins, vocabulary coverage, and citation presence—silently abstaining (`abstained`) with machine-readable reasons instead of hallucinating.

---

## 🏗 Architecture Flow

```
User Query
    │
    ├──▶ BM25 Index (Sparse Keyword Retrieval) ──────┐
    │                                                ▼
    └──▶ Dense Vector Index (sentence-transformers) ─▶ Reciprocal Rank Fusion (RRF, k=60)
                                                             │
                                                             ▼
                                                    Deterministic Gate
                                              (Coverage, Margin, Threshold)
                                                             │
                                        ┌────────────────────┴────────────────────┐
                                        ▼                                         ▼
                                   [Passed Gate]                            [Failed Gate]
                                Extractive Composer                        Safe Abstention
                             (Grounded Answer + Citations)             (Status: "abstained")
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install with Python 3.12+:

```bash
git clone https://github.com/NexosAILab/hybrid-rag-mcp-starter.git
cd hybrid-rag-mcp-starter
pip install -e .[local,dev]
```

### 2. Run the MCP Server (stdio)
```bash
python -m hybrid_rag_mcp.server
```

### 3. Register in Claude Code / Multi-Agent Coordinator (`mcp_servers.json`)
```json
{
  "mcpServers": {
    "knowledge": {
      "command": "python",
      "args": ["-m", "hybrid_rag_mcp.server"],
      "env": {
        "EMBEDDER": "local",
        "RRF_K": "60",
        "AS_OF": "2026-11-01"
      }
    }
  }
}
```

---

## 📊 Benchmark Results

Evaluation executed over the seeded corpus (48 documents, 24 labeled queries) across three configurations:

| Configuration | exact_id (6) | temporal (6) | paraphrase (6) | unanswerable (6) | overall (24) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **naive** | 4/6 | 2/6 | 6/6 | 0/6 | **12/24** |
| **hybrid** | 5/6 | 4/6 | 6/6 | 0/6 | **15/24** |
| **hybrid_gate** | 0/6 | 2/6 | 4/6 | 6/6 | **12/24** |

- **Embedder:** `local` (`sentence-transformers/all-MiniLM-L6-v2`)
- **Evaluation Date (`as_of`):** `2026-11-01`
- **Zero Hallucination:** On the 6 unanswerable questions, naive and un-gated hybrid answer every question from nearest neighbors. `hybrid_gate` abstains safely **6/6 times**.

---

## 🛠 Available MCP Tools

- `search_knowledge(query: str, top_k: int = 5)`: Returns top-ranked documents fused via BM25 and vector search with normalized scores and rank telemetry.
- `answer_with_citations(question: str)`: Returns a validated answer with cited document IDs or a structured abstention.
- `evaluate()`: Runs the 24-query offline benchmark suite live through the pipeline and returns category accuracy metrics.

---

## ⚖️ Limitations & Design Decisions

- **Extractive Grounding:** Answers are extracted directly from verified document sentences. An LLM generation layer can sit behind the gate, but the gate remains the deterministic arbiter of whether the context supports generation.
- **Grounding vs Truth:** The gate guarantees answers are grounded in the loaded corpus; it does not verify external real-world ground truth beyond the documents.

---

## 📄 License

MIT License. Built by [Nexos AI Lab](https://nexosailab.vercel.app).
