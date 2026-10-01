"""Run a small retrieval regression set against the current database.

Usage:
  python tools/eval_retrieval.py --database data/saga.db

The evaluator intentionally measures retrieval only.  It catches wrong
standard selection before an LLM answer or prompt change obscures the cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.database import Database  # noqa: E402
from saga.text import expand_with_synonyms  # noqa: E402


def run(database: Database, cases: list[dict], limit: int) -> dict:
    rows: list[dict] = []
    doc_hits = 0
    section_hits = 0
    for case in cases:
        results = database.hybrid_search(
            expand_with_synonyms(str(case["query"])),
            limit,
            list(case.get("doc_codes") or []) or None,
            "CODE" if case.get("doc_codes") else "CROSS",
        )
        expected_codes = {str(code).upper() for code in case.get("doc_codes", [])}
        code_hit = bool(results) and any(
            str(item.get("doc_code", "")).upper() in expected_codes for item in results
        )
        section = str(case.get("section", ""))
        section_hit = code_hit and any(section in str(item.get("hierarchy", "")) for item in results)
        doc_hits += int(code_hit)
        section_hits += int(section_hit)
        rows.append(
            {
                "id": case.get("id", case["query"]),
                "doc_hit": code_hit,
                "section_hit": section_hit,
                "top": [
                    {
                        "doc_code": item.get("doc_code"),
                        "page": item.get("page"),
                        "hierarchy": item.get("hierarchy"),
                    }
                    for item in results[:3]
                ],
            }
        )
    total = max(1, len(cases))
    return {
        "cases": len(cases),
        "document_recall_at_k": round(doc_hits / total, 4),
        "section_recall_at_k": round(section_hits / total, 4),
        "k": limit,
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "retrieval_cases.jsonl")
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    database = Database(args.database)
    database.initialize()
    print(json.dumps(run(database, cases, max(3, args.k)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
