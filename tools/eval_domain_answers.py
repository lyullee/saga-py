"""Run the generated law/code/rule question bank through the real RAG pipeline.

This is deliberately serial: the Service Hub development key permits one
concurrent request. Results are JSONL so a long run can be resumed or audited
without exposing credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.config import get_settings  # noqa: E402
from saga.database import Database  # noqa: E402
from saga.rag import RagPipeline  # noqa: E402
from saga.schemas import ChatRequest  # noqa: E402
from saga.service_hub_client import ServiceHubReasoner  # noqa: E402


def _quality_flags(answer: str, citations: list[dict], domain: str) -> list[str]:
    text = answer or ""
    flags: list[str] = []
    if len(text) < 120 and domain != "GENERAL":
        flags.append("too_short")
    if "답변을 보류" in text or "답변드리기 어렵" in text:
        flags.append("refusal")
    if domain in {"LAW", "CODE", "RULE"} and not citations:
        flags.append("no_citation")
    if citations and not any(f"[{i}]" in text for i in range(1, len(citations) + 1)):
        flags.append("citation_marker_missing")
    if text.count("[1]") >= 4 and len(text) < 900 and not re.search(r"(?m)^\s*\d+[.)]", text):
        flags.append("citation_dump_risk")
    return flags


async def evaluate(
    cases: list[dict],
    output: Path,
    limit_per_domain: int | None,
    start: int = 0,
    pause_seconds: float = 0.0,
) -> dict:
    settings = get_settings()
    database = Database(settings.database_path)
    database.initialize()
    reasoner = ServiceHubReasoner(settings)
    pipeline = RagPipeline(settings, database, reasoner)
    counts = Counter()
    flagged = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    per_domain = Counter()
    for case in cases:
        domain = str(case.get("domain", ""))
        if limit_per_domain is not None and per_domain[domain] >= limit_per_domain:
            continue
        per_domain[domain] += 1
        selected.append(case)

    # append mode permits a stopped run to be continued with --start.
    mode = "a" if start else "w"
    with output.open(mode, encoding="utf-8") as fp:
        for index, case in enumerate(selected[start:], start=start):
            domain = str(case.get("domain", ""))
            query = str(case["query"])
            started = time.perf_counter()
            try:
                request = ChatRequest(
                    message=query,
                    conversation_id=f"eval-{case['id']}",
                    answer_length="standard",
                )
                response = await pipeline.run(request)
                answer = response.answer
                citations = [item.model_dump() for item in response.citations]
                row = {
                    "index": index,
                    "id": case["id"],
                    "domain": domain,
                    "query": query,
                    "expected_doc_codes": case.get("expected_doc_codes", []),
                    "answer_mode": response.answer_mode,
                    "model": response.model,
                    "intent": response.intent,
                    "rewritten_query": response.rewritten_query,
                    "answer": answer,
                    "citations": citations,
                    "quality_flags": _quality_flags(answer, citations, domain),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            except Exception as exc:  # keep the bank auditable on one bad case
                row = {
                    "index": index,
                    "id": case["id"],
                    "domain": domain,
                    "query": query,
                    "expected_doc_codes": case.get("expected_doc_codes", []),
                    "error": f"{type(exc).__name__}: {exc}",
                    "quality_flags": ["error"],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            counts[domain] += 1
            for flag in row.get("quality_flags", []):
                flagged[f"{domain}:{flag}"] += 1
            print(json.dumps({"done": index + 1, "id": case["id"], "domain": domain, "flags": row.get("quality_flags", [])}, ensure_ascii=False), flush=True)
            if pause_seconds > 0 and index + 1 < len(selected[start:]):
                await asyncio.sleep(pause_seconds)
    await reasoner.close()
    return {"processed": sum(counts.values()), "counts": dict(counts), "flags": dict(flagged), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "domain_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "domain_answers.jsonl")
    parser.add_argument("--limit-per-domain", type=int, default=10)
    parser.add_argument("--start", type=int, default=0, help="skip this many selected cases when resuming")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="pause between cases to respect RPM limits")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = asyncio.run(evaluate(cases, args.output, args.limit_per_domain, args.start, args.pause_seconds))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
