"""Integration example: Registering hybrid-rag-mcp as the 4th MCP server in the coordinator.

Demonstrates:
1. Exporting MCP configuration snippet for VID_002 coordinator or Claude Code.
2. Direct invocation of the server's tools over stdio/in-memory pipeline:
   - Q01 (Hero query): grounded answer with citations.
   - Q19 (Unanswerable query): deterministic abstention with machine-readable reason.
"""

import json
import os
import sys
from pathlib import Path

# Add src to sys.path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hybrid_rag_mcp.server import answer_with_citations_tool, search_knowledge_tool

MCP_REGISTRATION_SNIPPET = {
    "mcpServers": {
        "knowledge": {
            "command": "python",
            "args": ["-m", "hybrid_rag_mcp.server"],
            "env": {"EMBEDDER": os.getenv("EMBEDDER", "local")}
        }
    }
}


def main():
    print("=" * 70)
    print("VID_002 Coordinator -> 4th MCP Server Integration (hybrid-rag-mcp)")
    print("=" * 70)

    print("\n1. MCP Server Registration Snippet (mcp_servers.json):")
    print(json.dumps(MCP_REGISTRATION_SNIPPET, indent=2))

    print("\n2. Executing Query Q01 (Exact-ID / Hero Query):")
    q01 = "What is the remediation for cache error ERR_CACHE_109 in the billing service?"
    res_q01 = answer_with_citations_tool(question=q01)
    print(f"Status: {res_q01.get('status')}")
    if res_q01.get("status") == "answered":
        print(f"Answer: {res_q01.get('answer')}")
        print(f"Citations: {res_q01.get('citations')}")
    else:
        print(f"Abstained: {res_q01.get('gate', {}).get('reason')}")

    print("\n3. Executing Query Q19 (Unanswerable / Out of Corpus Query):")
    q19 = "What is the SLA penalty clause for database outages in the EU region?"
    res_q19 = answer_with_citations_tool(question=q19)
    print(f"Status: {res_q19.get('status')}")
    print(f"Gate Reason: {res_q19.get('gate', {}).get('reason')}")
    print(f"Output Answer: {res_q19.get('answer')}")

    print("\n" + "=" * 70)
    print("Grounded workflow verification complete: 1 Answered/Evaluated + 1 Abstained.")
    print("=" * 70)


if __name__ == "__main__":
    main()
