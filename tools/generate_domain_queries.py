"""Generate broad, reproducible domain QA cases from the indexed corpus.

The cases are intentionally phrased as user questions rather than keyword
queries.  They are used both for retrieval regression and for optional LLM
answer review (``tools/eval_domain_answers.py``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.database import Database  # noqa: E402
from saga.law_api import LAW_CATALOG  # noqa: E402


LAW_TEMPLATES = (
    "{name}의 목적과 적용 대상은 무엇인가요?",
    "{name}에서 정한 사업자 또는 시설의 기본 의무를 설명해줘.",
    "{name}의 안전관리자 선임·업무와 책임을 정리해줘.",
    "{name}에 따른 허가·신고·변경 절차를 구분해서 알려줘.",
    "{name}에서 시설을 설치하거나 변경할 때 확인할 기준은 무엇인가요?",
    "{name}의 검사·점검·안전관리 절차를 현장 순서로 설명해줘.",
    "{name}에서 사고나 누출이 발생했을 때 사업자가 취해야 할 조치는 무엇인가요?",
    "{name}의 기록·보고·보존 의무를 실무자가 이해하기 쉽게 설명해줘.",
    "{name}에서 규정한 사업 중지·개선명령·행정처분의 조건을 알려줘.",
    "{name} 관련 위반 시 책임과 제재를 설명하되 조문 근거가 있는 범위만 답해줘.",
)

CODE_TEMPLATES = (
    "KGS {code}의 적용범위와 대상 시설을 설명해줘.",
    "KGS {code}에서 요구하는 주요 설치 기준을 단계별로 정리해줘.",
    "KGS {code}의 재료·구조·압력 조건 중 현장에서 놓치기 쉬운 부분은 무엇인가요?",
    "KGS {code}에 따른 검사 종류와 검사 시점을 구분해줘.",
    "KGS {code}의 기밀·내압·누출 관련 시험 조건을 설명해줘.",
    "KGS {code}에서 정한 예외·면제·제외 조건이 있다면 함께 알려줘.",
    "KGS {code}를 기준으로 작업 전 확인할 체크리스트를 만들어줘.",
    "KGS {code}의 안전거리·방호·차단 요구사항을 이해하기 쉽게 풀어줘.",
    "KGS {code}에서 수치가 나오는 조항을 찾아 단위와 조건을 그대로 설명해줘.",
    "KGS {code}와 유사한 시설에 다른 기준이 적용될 때 경계를 어떻게 판단하나요?",
)

RULE_TEMPLATES = (
    "사규 {code}의 목적과 적용 범위를 요약해줘.",
    "사규 {code}에서 담당자와 승인권자의 역할을 구분해줘.",
    "사규 {code}의 업무 처리 절차를 처음부터 끝까지 설명해줘.",
    "사규 {code}에서 제출해야 하는 서류와 기록을 알려줘.",
    "사규 {code}의 검사·점검·심사 기준을 현장 실무 관점에서 정리해줘.",
    "사규 {code}에서 예외나 면제 조건을 어떻게 다루는지 알려줘.",
    "사규 {code}에 따른 보고·통보 기한과 후속 조치를 설명해줘.",
    "사규 {code}를 준수하지 않았을 때 보완·재검토 절차는 무엇인가요?",
    "사규 {code}의 핵심 내용을 신입 직원도 이해할 수 있게 풀어줘.",
    "사규 {code}와 관련된 안전관리 업무를 체크리스트로 만들어줘.",
)


def _pick_codes(rows: list[dict], preferred: list[str], count: int) -> list[dict]:
    by_code = {str(row["doc_code"]): row for row in rows}
    selected = [by_code[code] for code in preferred if code in by_code]
    if len(selected) < count:
        selected.extend(row for row in rows if row not in selected)
    return selected[:count]


def generate(database: Database) -> list[dict]:
    rows = database.list_documents()
    codes = [row for row in rows if row["doc_type"] == "CODE"]
    rules = [row for row in rows if row["doc_type"] == "RULE"]
    # These span hydrogen, city-gas, LPG, high-pressure gas, inspection and
    # storage topics.  If a corpus edition lacks one, the fallback picker keeps
    # the set at 120 cases while preserving valid expected codes.
    code_targets = _pick_codes(
        codes,
        [
            "FS551", "FU671", "FU551", "FP217", "FP216", "FP111", "FP112", "FP113",
            "FP211", "FP333", "AC111", "AC118", "AH271", "AH371", "AA915", "AA632",
        ],
        12,
    )
    rule_targets = _pick_codes(
        rules,
        [
            "2400-1", "2400-2", "2401-1", "2302-1", "2300-20", "2301-1", "2100-1",
            "2100-4", "2100-5", "2101-5", "2101-6", "2201-1", "2201-2", "2500-1",
        ],
        12,
    )
    cases: list[dict] = []
    sequence = 1
    for abbreviation, name in LAW_CATALOG.items():
        for template in LAW_TEMPLATES:
            cases.append({
                "id": f"law-{sequence:03d}",
                "domain": "LAW",
                "category": "law",
                "query": template.format(name=name),
                "expected_doc_codes": [f"LAW-{name}"],
                "expected_terms": [name],
                "source": "generated-law-catalog",
            })
            sequence += 1
    sequence = 1
    for target in code_targets:
        code = str(target["doc_code"])
        for template in CODE_TEMPLATES:
            cases.append({
                "id": f"code-{sequence:03d}",
                "domain": "CODE",
                "category": "code",
                "query": template.format(code=code),
                "expected_doc_codes": [code],
                "expected_terms": [code, str(target["title"])[:50]],
                "source": "generated-indexed-code",
            })
            sequence += 1
    sequence = 1
    for target in rule_targets:
        code = str(target["doc_code"])
        for template in RULE_TEMPLATES:
            cases.append({
                "id": f"rule-{sequence:03d}",
                "domain": "RULE",
                "category": "rule",
                "query": template.format(code=code),
                "expected_doc_codes": [code],
                "expected_terms": [code, str(target["title"])[:50]],
                "source": "generated-indexed-rule",
            })
            sequence += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "domain_queries.jsonl")
    args = parser.parse_args()
    database = Database(args.database)
    database.initialize()
    cases = generate(database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    counts = {category: sum(case["category"] == category for case in cases) for category in ("law", "code", "rule")}
    print(json.dumps({"total": len(cases), "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
