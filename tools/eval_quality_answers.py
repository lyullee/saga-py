"""Evaluate answer quality across RAG, LLM fallback, clarification and noise cases."""

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
from saga.rag import LLM_LIMITED_NOTICE, LLM_ONLY_NOTICE, RagPipeline  # noqa: E402
from saga.schemas import ChatRequest  # noqa: E402
from saga.service_hub_client import ServiceHubReasoner  # noqa: E402


def _has_disclosure(text: str) -> bool:
    return any(
        marker in (text or "")
        for marker in (LLM_ONLY_NOTICE, LLM_LIMITED_NOTICE, "LLM 자체 판단", "일반 지식")
    )


def _quality_flags(case: dict, answer: str, citations: list[dict], mode: str) -> list[str]:
    category = str(case.get("quality_category", ""))
    text = answer or ""
    expected_modes = set(case.get("expected_answer_mode", []))
    flags: list[str] = []
    if expected_modes and mode not in expected_modes:
        flags.append("wrong_answer_mode")
    if category in {"rag_direct", "llm_fallback"} and len(text) < int(case.get("target_chars", 320)):
        flags.append("too_short")
    if re.search(r"답변을\s*(?:보류|드리기\s*어렵)|추가\s*확인이\s*필요합니다\s*$", text):
        flags.append("refusal_or_dead_end")
    if category == "rag_direct":
        # A RAG-first case is allowed to fall back when the retrieved chunks
        # do not contain the requested legal/technical topic.  In that mode
        # the disclosure and absence of misleading citations are the quality
        # checks; requiring a document code would mark the honest result as a
        # failure.
        if mode == "llm_only" and "llm_only" in expected_modes:
            if not _has_disclosure(text):
                flags.append("fallback_disclosure_missing")
            if citations:
                flags.append("fallback_has_misleading_citations")
            return flags
        expected_codes = {str(code).upper() for code in case.get("expected_doc_codes", [])}
        actual_codes = {str(item.get("doc_code", "")).upper() for item in citations}
        if not citations:
            flags.append("no_rag_citation")
        if expected_codes and not expected_codes.intersection(actual_codes):
            flags.append("wrong_source_document")
        if citations and not any(f"[{number}]" in text for number in range(1, len(citations) + 1)):
            flags.append("citation_marker_missing")
        if len(citations) >= 3 and len(re.findall(r"\[\d+\]", text)) >= len(citations) and not re.search(
            r"핵심|결론|따라서|의미|조건|예외|확인", text[:500]
        ):
            flags.append("search_result_dump")
    elif category == "llm_fallback":
        if mode == "clarification":
            flags.append("fallback_stopped_as_clarification")
        if mode == "llm_only" and not _has_disclosure(text):
            flags.append("fallback_disclosure_missing")
        if mode == "llm_only" and citations:
            flags.append("fallback_has_misleading_citations")
    elif category == "off_topic":
        if mode != "llm_only":
            flags.append("off_topic_not_llm_only")
        if citations:
            flags.append("off_topic_has_citation")
        if not _has_disclosure(text):
            flags.append("off_topic_disclosure_missing")
    elif category == "clarification":
        # Strictly underspecified prompts should ask for scope.  Concrete but
        # terse prompts may legitimately be answered with a qualified RAG or
        # LLM fallback, so only reject citations when the selected mode is
        # actually clarification.
        if mode == "clarification" and citations:
            flags.append("clarification_has_citation")
        if mode == "llm_only" and not _has_disclosure(text):
            flags.append("fallback_disclosure_missing")
    return flags


async def evaluate(
    cases: list[dict],
    output: Path,
    limit_per_category: int | None,
    pause_seconds: float,
    answer_length: str,
) -> dict:
    settings = get_settings()
    database = Database(settings.database_path)
    database.initialize()
    reasoner = ServiceHubReasoner(settings)
    pipeline = RagPipeline(settings, database, reasoner)
    selected: list[dict] = []
    category_counts: Counter[str] = Counter()
    for case in cases:
        category = str(case.get("quality_category", ""))
        if limit_per_category is not None and category_counts[category] >= limit_per_category:
            continue
        category_counts[category] += 1
        selected.append(case)

    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as fp:
        for index, case in enumerate(selected):
            started = time.perf_counter()
            try:
                request = ChatRequest(
                    message=str(case["query"]),
                    conversation_id=f"quality-{case['id']}",
                    answer_length=answer_length,
                )
                response = await pipeline.run(request)
                answer = response.answer
                citations = [item.model_dump() for item in response.citations]
                row = {
                    "index": index,
                    "id": case["id"],
                    "quality_category": case.get("quality_category"),
                    "query": case["query"],
                    "expected_doc_codes": case.get("expected_doc_codes", []),
                    "expected_answer_mode": case.get("expected_answer_mode", []),
                    "answer_mode": response.answer_mode,
                    "model": response.model,
                    "intent": response.intent,
                    "answer_length_setting": answer_length,
                    "answer_chars": len(answer or ""),
                    "answer": answer,
                    "citations": citations,
                    "quality_flags": _quality_flags(case, answer, citations, response.answer_mode),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            except Exception as exc:
                row = {
                    "index": index,
                    "id": case["id"],
                    "quality_category": case.get("quality_category"),
                    "query": case["query"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "quality_flags": ["error"],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            category = str(row.get("quality_category", ""))
            counts[category] += 1
            for flag in row.get("quality_flags", []):
                flags[f"{category}:{flag}"] += 1
            print(
                json.dumps(
                    {"done": index + 1, "total": len(selected), "id": row["id"], "flags": row.get("quality_flags", [])},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if pause_seconds > 0 and index + 1 < len(selected):
                await asyncio.sleep(pause_seconds)
    await reasoner.close()
    summary = {"processed": sum(counts.values()), "counts": dict(counts), "flags": dict(flags), "output": str(output)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "quality_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "quality_answers.jsonl")
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--pause-seconds", type=float, default=6.5)
    parser.add_argument("--answer-length", choices=("standard", "detailed", "very_detailed"), default="detailed")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = asyncio.run(evaluate(cases, args.output, args.limit_per_category, args.pause_seconds, args.answer_length))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
