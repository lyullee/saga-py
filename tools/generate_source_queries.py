"""Generate a 300-question retrieval regression bank.

The bank is intentionally split into three user-facing knowledge modes:

* 100 standards/law/rule questions
* 100 NREL operations and failure-data questions
* 100 HIAD incident-case questions

The questions are deterministic so that a later retrieval or prompt change can
be compared against the same corpus.  Document identifiers are selected from
the currently indexed database instead of being hard-coded to one installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from saga.database import Database  # noqa: E402


STANDARD_TEMPLATES = (
    "{label}의 적용 범위와 대상 시설을 설명해줘.",
    "{label}에서 정한 핵심 안전관리 의무를 현장 관점에서 풀어줘.",
    "{label}에 따라 설치 또는 변경 전에 확인할 항목을 단계별로 정리해줘.",
    "{label}의 검사·점검·시험 종류와 시행 시점을 구분해줘.",
    "{label}에서 수치나 조건이 정해진 부분을 근거와 함께 설명해줘.",
    "{label}의 예외·제외·면제 조건이 있다면 함께 알려줘.",
    "{label}을 기준으로 작업 전 체크리스트를 만들어줘.",
    "{label}의 기록·보고·보존 의무를 실무자가 이해하기 쉽게 설명해줘.",
    "{label}을 지키지 못했을 때 보완 또는 후속 절차를 알려줘.",
    "{label}의 내용을 신입 안전관리자도 이해할 수 있게 핵심부터 정리해줘.",
)

NREL_TOPICS = (
    ("충전소 배치와 운영 기간", ("deployment", "stations", "timeline")),
    ("분기별 수소 공급량과 충전 횟수", ("hydrogen dispensed", "fills", "quarter")),
    ("충전 속도·충전 시간·충전량", ("fueling rate", "fueling time", "amount")),
    ("350 bar와 700 bar 충전 성능", ("350 bar", "700 bar", "fueling rate")),
    ("충전 사이 시간과 최종 압력", ("time between fueling", "final pressure")),
    ("충전소 용량 이용률과 실제 사용량", ("capacity utilization", "nameplate capacity")),
    ("일별·시간대별 충전 수요", ("fills per day", "time of day")),
    ("수소 가격과 충전 지역별 공급량", ("hydrogen price", "region", "dispensed")),
    ("수소 품질과 SAE J2719 불순물 기준", ("hydrogen quality", "impurities", "SAE J2719")),
    ("NREL 자료에서 안전·신뢰성·유지보수 데이터의 공개 범위", ("safety", "reliability", "maintenance")),
)

NREL_TEMPLATES = (
    "NREL 자료에서 {topic}을 찾아 핵심 결과를 설명해줘.",
    "NREL 충전소 운영 데이터로 {topic}의 추세를 알려줘.",
    "{topic}에 대해 NREL 보고서가 실제로 보여주는 내용은 무엇인가요?",
    "NREL CDP에서 {topic}을 확인하고 현장 운영에 어떤 의미인지 해석해줘.",
    "수소충전소 디지털 트윈에 활용할 수 있도록 {topic}을 정리해줘.",
    "{topic}과 관련된 NREL의 수치·기간·조건을 근거와 함께 알려줘.",
    "NREL 보고서를 기준으로 {topic}을 고장 예방 관점에서 설명해줘.",
    "{topic}에 대해 NREL 데이터의 한계와 해석 시 주의점도 함께 알려줘.",
    "운전·고장 분석을 위해 NREL 자료에서 {topic}을 어떻게 읽어야 하나요?",
    "{topic}에 관한 NREL 근거를 찾아 쉬운 한국어로 설명해줘.",
)

HIAD_TOPICS = (
    ("수소 누출·누설 사건", ("hydrogen", "leak", "release")),
    ("화재 사건", ("hydrogen", "fire")),
    ("폭발 사건", ("hydrogen", "explosion")),
    ("압축기 관련 사고", ("compressor", "incident")),
    ("디스펜서·노즐 관련 사고", ("dispenser", "nozzle")),
    ("저장탱크·저장용기 사고", ("storage", "tank")),
    ("배관·밸브 관련 사고", ("pipeline", "valve")),
    ("사고의 근본 원인", ("root cause", "cause")),
    ("인명 피해·설비 손상 결과", ("injury", "damage", "consequence")),
    ("사고 교훈·예방·비상 대응", ("lesson", "prevention", "response")),
)

HIAD_TEMPLATES = (
    "HIAD 사례에서 {topic}을 찾아 대표적인 사실을 설명해줘.",
    "HIAD 2.2 데이터에 기록된 {topic}의 공통 양상을 정리해줘.",
    "수소충전소 사고 사례 중 {topic}에 해당하는 기록을 알려줘.",
    "{topic}과 관련해 HIAD 원문에서 확인되는 원인과 결과를 구분해줘.",
    "디지털 트윈 사고 시나리오에 활용할 수 있도록 {topic} 사례를 요약해줘.",
    "HIAD에서 {topic}의 발생 장치와 상황을 찾아 설명해줘.",
    "{topic}에 관한 HIAD 사례를 근거 번호와 함께 비교해줘.",
    "HIAD 기록으로 {topic}을 분석할 때 일반화하면 안 되는 한계도 알려줘.",
    "사고 대응 훈련을 위해 HIAD의 {topic} 사례에서 배울 점을 정리해줘.",
    "{topic}에 대한 HIAD 공개 기록을 쉬운 한국어로 해석해줘.",
)


def _pick(rows: list[dict], doc_type: str, preferred: list[str], count: int) -> list[dict]:
    candidates = [row for row in rows if row.get("doc_type") == doc_type]
    by_code = {str(row.get("doc_code")): row for row in candidates}
    selected = [by_code[code] for code in preferred if code in by_code]
    selected.extend(row for row in candidates if row not in selected)
    return selected[:count]


def _standard_cases(database: Database) -> list[dict]:
    rows = database.list_documents()
    code_targets = _pick(rows, "CODE", ["FS551", "FU671", "FP217", "FP216"], 4)
    law_targets = _pick(
        rows,
        "LAW",
        [
            "LAW-수소경제 육성 및 수소 안전관리에 관한 법률",
            "LAW-고압가스 안전관리법",
            "LAW-액화석유가스의 안전관리 및 사업법",
        ],
        3,
    )
    rule_targets = _pick(rows, "RULE", ["2400-1", "2201-1", "2100-1"], 3)
    groups = [("code", code_targets), ("law", law_targets), ("rule", rule_targets)]
    cases: list[dict] = []
    sequence = 1
    for category, targets in groups:
        for target in targets:
            code = str(target["doc_code"])
            label = code if category != "law" else code.removeprefix("LAW-")
            for template in STANDARD_TEMPLATES:
                cases.append(
                    {
                        "id": f"standards-{sequence:03d}",
                        "knowledge_mode": "standards",
                        "domain": str(target["doc_type"]),
                        "category": category,
                        "query": template.format(label=label),
                        "expected_doc_codes": [code],
                        "expected_source_types": [str(target["doc_type"])],
                        "expected_terms": list(dict.fromkeys((code, label))),
                        "source": "indexed-standards-corpus",
                    }
                )
                sequence += 1
    if len(cases) != 100:
        raise RuntimeError(f"standards target selection produced {len(cases)} cases, expected 100")
    return cases


def _external_cases(
    mode: str,
    topics: tuple[tuple[str, tuple[str, ...]], ...],
    templates: tuple[str, ...],
    doc_type: str,
    doc_code: str,
) -> list[dict]:
    cases: list[dict] = []
    sequence = 1
    for topic, terms in topics:
        for template in templates:
            cases.append(
                {
                    "id": f"{mode}-{sequence:03d}",
                    "knowledge_mode": mode,
                    "domain": "CROSS",
                    "category": mode,
                    "query": template.format(topic=topic),
                    "expected_doc_codes": [doc_code],
                    "expected_source_types": [doc_type],
                    "expected_terms": list(terms),
                    "source": "indexed-public-source",
                }
            )
            sequence += 1
    if len(cases) != 100:
        raise RuntimeError(f"{mode} produced {len(cases)} cases, expected 100")
    return cases


def generate(database: Database) -> list[dict]:
    cases = _standard_cases(database)
    cases.extend(_external_cases("operations", NREL_TOPICS, NREL_TEMPLATES, "NREL", "NREL-CDP"))
    cases.extend(_external_cases("incidents", HIAD_TOPICS, HIAD_TEMPLATES, "HIAD", "HIAD-2.2"))
    if len(cases) != 300:
        raise RuntimeError(f"generated {len(cases)} cases, expected 300")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "source_queries_300.jsonl")
    args = parser.parse_args()
    database = Database(args.database)
    database.initialize()
    cases = generate(database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    counts = {mode: sum(case["knowledge_mode"] == mode for case in cases) for mode in ("standards", "operations", "incidents")}
    print(json.dumps({"total": len(cases), "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
