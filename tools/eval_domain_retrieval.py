"""Evaluate document selection for the generated law/code/rule question bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.database import Database  # noqa: E402
from saga.text import expand_with_synonyms  # noqa: E402


def evaluate(database: Database, cases: list[dict], limit: int = 12, filter_expected: bool = False) -> dict:
    details: list[dict] = []
    stats = defaultdict(lambda: Counter())
    for case in cases:
        domain = str(case["domain"])
        expected = {str(code).upper() for code in case.get("expected_doc_codes", [])}
        results = database.hybrid_search(
            expand_with_synonyms(str(case["query"])),
            max(3, limit),
            list(expected) if filter_expected and domain in {"CODE", "RULE", "LAW"} else None,
            domain,
        )
        returned = [str(item.get("doc_code", "")).upper() for item in results]
        hit = bool(expected.intersection(returned))
        top1 = bool(returned and returned[0] in expected)
        stats[domain]["total"] += 1
        stats[domain]["hit"] += int(hit)
        stats[domain]["top1"] += int(top1)
        details.append({
            "id": case["id"],
            "domain": domain,
            "query": case["query"],
            "expected": sorted(expected),
            "hit": hit,
            "top1": top1,
            "results": [
                {"doc_type": item.get("doc_type"), "doc_code": item.get("doc_code"), "title": item.get("title"), "page": item.get("page")}
                for item in results[:3]
            ],
        })
    summary = {}
    for domain, counter in stats.items():
        total = max(1, counter["total"])
        summary[domain] = {
            "cases": counter["total"],
            "document_recall_at_k": round(counter["hit"] / total, 4),
            "top1_recall": round(counter["top1"] / total, 4),
            "k": limit,
        }
    return {"summary": summary, "cases": len(cases), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "domain_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "domain_retrieval_report.json")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--filter-expected", action="store_true", help="문서번호를 SQL 필터로 고정해 조항 검색만 평가")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    database = Database(args.database)
    database.initialize()
    report = evaluate(database, cases, args.k, args.filter_expected)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
