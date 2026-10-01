"""Measure retrieval quality and source isolation for the 300-question bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.database import Database  # noqa: E402
from saga.external_sources import expand_external_query  # noqa: E402
from saga.text import expand_with_synonyms, normalize_text  # noqa: E402


def _term_hit(case: dict, results: list[dict]) -> bool:
    terms = [normalize_text(str(term)).lower() for term in case.get("expected_terms", []) if normalize_text(str(term))]
    haystack = " ".join(
        normalize_text(" ".join(str(item.get(key, "")) for key in ("title", "hierarchy", "content"))).lower()
        for item in results
    )
    return any(term in haystack for term in terms)


def evaluate(database: Database, cases: list[dict], limit: int = 12, expand_external: bool = False) -> dict:
    details: list[dict] = []
    stats: defaultdict[str, Counter] = defaultdict(Counter)
    for case in cases:
        mode = str(case["knowledge_mode"])
        domain = str(case.get("domain", "CROSS"))
        expected_codes = {str(code).upper() for code in case.get("expected_doc_codes", [])}
        source_types = [str(value).upper() for value in case.get("expected_source_types", [])] or None
        query = expand_with_synonyms(str(case["query"]))
        if source_types:
            # Evaluate the same source-aware expansion used by the runtime.
            # The flag is retained to produce a directly comparable baseline
            # report for regression tracking.
            query = expand_external_query(
                str(case["query"]) if expand_external else query,
                source_types[0],
            )
        results = database.hybrid_search(query, max(3, limit), None, domain, source_types)
        returned_codes = [str(item.get("doc_code", "")).upper() for item in results]
        returned_types = [str(item.get("doc_type", "")).upper() for item in results]
        hit = bool(expected_codes.intersection(returned_codes))
        top1 = bool(returned_codes and returned_codes[0] in expected_codes)
        term_hit = _term_hit(case, results)
        pure = bool(results) and all(value in set(source_types or returned_types) for value in returned_types)
        stats[mode]["total"] += 1
        stats[mode]["hit"] += int(hit)
        stats[mode]["top1"] += int(top1)
        stats[mode]["term_hit"] += int(term_hit)
        stats[mode]["source_pure"] += int(pure)
        details.append(
            {
                "id": case["id"],
                "knowledge_mode": mode,
                "category": case.get("category"),
                "query": case["query"],
                "expanded_query": query,
                "expected_doc_codes": sorted(expected_codes),
                "expected_source_types": source_types or [],
                "hit": hit,
                "top1": top1,
                "term_hit": term_hit,
                "source_pure": pure,
                "results": [
                    {
                        "doc_type": item.get("doc_type"),
                        "doc_code": item.get("doc_code"),
                        "title": item.get("title"),
                        "page": item.get("page"),
                        "score": item.get("score"),
                    }
                    for item in results[:5]
                ],
            }
        )
    summary: dict[str, dict] = {}
    for mode, counter in stats.items():
        total = max(1, counter["total"])
        summary[mode] = {
            "cases": counter["total"],
            "document_recall_at_k": round(counter["hit"] / total, 4),
            "top1_recall": round(counter["top1"] / total, 4),
            "term_hit_at_k": round(counter["term_hit"] / total, 4),
            "source_purity": round(counter["source_pure"] / total, 4),
            "k": limit,
        }
    return {"summary": summary, "cases": len(cases), "expanded_external": expand_external, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "source_queries_300.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "source_queries_300_report.json")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--expand-external", action="store_true")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    database = Database(args.database)
    database.initialize()
    report = evaluate(database, cases, args.k, args.expand_external)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
