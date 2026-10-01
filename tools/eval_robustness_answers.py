"""Run noisy/out-of-domain questions through the real pipeline.

The script is intentionally serial because the development Service Hub key is
limited to one concurrent request.  It records the complete answer and the
expected-vs-actual routing decision so a failed case can be reproduced without
guessing from a screenshot.
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
from saga.rag import LLM_ONLY_NOTICE, RagPipeline  # noqa: E402
from saga.schemas import ChatRequest  # noqa: E402
from saga.service_hub_client import ServiceHubReasoner  # noqa: E402


PROFANITY_OR_SLANG = re.compile(r"ㅈㄴ|개빡|어캄|어케|뭐임|몇이냐|해야됨|함\?|존나|개같", re.I)


def _quality_flags(case: dict, answer: str, citations: list[dict], answer_mode: str) -> list[str]:
    expected = set(case.get("expected_answer_mode", []))
    text = answer or ""
    flags: list[str] = []
    if answer_mode not in expected:
        flags.append("wrong_answer_mode")
    if case.get("expected_behavior") == "recover_rag" and not citations:
        flags.append("no_citation")
    if case.get("expected_behavior") == "llm_only":
        if citations:
            flags.append("citation_on_off_topic")
        if LLM_ONLY_NOTICE not in text:
            flags.append("llm_only_notice_missing")
    if answer_mode == "rag" and citations and not any(f"[{i}]" in text for i in range(1, len(citations) + 1)):
        flags.append("citation_marker_missing")
    if case.get("expected_behavior") == "recover_rag" and len(text) < 120:
        flags.append("too_short")
    if "답변을 보류" in text or "답변드리기 어렵" in text:
        flags.append("refusal")
    if PROFANITY_OR_SLANG.search(text):
        flags.append("slang_repeated")
    return flags


async def evaluate(
    cases: list[dict],
    output: Path,
    limit_per_category: int | None,
    start: int = 0,
    pause_seconds: float = 0.0,
) -> dict:
    settings = get_settings()
    database = Database(settings.database_path)
    database.initialize()
    reasoner = ServiceHubReasoner(settings)
    pipeline = RagPipeline(settings, database, reasoner)
    selected: list[dict] = []
    per_category = Counter()
    for case in cases:
        category = str(case.get("category", ""))
        if limit_per_category is not None and per_category[category] >= limit_per_category:
            continue
        per_category[category] += 1
        selected.append(case)

    counts = Counter()
    flagged = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start else "w"
    with output.open(mode, encoding="utf-8") as fp:
        for index, case in enumerate(selected[start:], start=start):
            category = str(case.get("category", ""))
            query = str(case["query"])
            started = time.perf_counter()
            try:
                request = ChatRequest(
                    message=query,
                    conversation_id=f"robust-{case['id']}",
                    answer_length="standard",
                )
                response = await pipeline.run(request)
                answer = response.answer
                citations = [item.model_dump() for item in response.citations]
                row = {
                    "index": index,
                    "id": case["id"],
                    "category": category,
                    "query": query,
                    "expected_behavior": case.get("expected_behavior"),
                    "expected_answer_mode": case.get("expected_answer_mode", []),
                    "answer_mode": response.answer_mode,
                    "model": response.model,
                    "intent": response.intent,
                    "rewritten_query": response.rewritten_query,
                    "answer": answer,
                    "citations": citations,
                    "quality_flags": _quality_flags(case, answer, citations, response.answer_mode),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            except Exception as exc:  # keep the bank auditable on one bad case
                row = {
                    "index": index,
                    "id": case["id"],
                    "category": category,
                    "query": query,
                    "expected_behavior": case.get("expected_behavior"),
                    "expected_answer_mode": case.get("expected_answer_mode", []),
                    "error": f"{type(exc).__name__}: {exc}",
                    "quality_flags": ["error"],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            counts[category] += 1
            for flag in row.get("quality_flags", []):
                flagged[f"{category}:{flag}"] += 1
            print(
                json.dumps(
                    {"done": index + 1, "id": case["id"], "category": category, "flags": row.get("quality_flags", [])},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if pause_seconds > 0 and index + 1 < len(selected[start:]):
                await asyncio.sleep(pause_seconds)
    await reasoner.close()
    return {
        "processed": sum(counts.values()),
        "counts": dict(counts),
        "flags": dict(flagged),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "robustness_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "robustness_answers.jsonl")
    parser.add_argument("--limit-per-category", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = asyncio.run(
        evaluate(cases, args.output, args.limit_per_category, args.start, args.pause_seconds)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
