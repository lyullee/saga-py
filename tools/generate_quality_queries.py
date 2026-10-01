"""Build a second-generation answer-quality bank for SAGA.

The older domain bank mostly checked whether the expected document was found.
This bank adds cases that force the answer layer to prove four behaviors:

1. use the local PDF/RAG evidence first;
2. disclose the boundary and answer from general LLM knowledge when evidence is
   missing or does not directly support the question;
3. produce a useful explanation rather than a short title/list response; and
4. keep ambiguous and unrelated questions out of the grounded route.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FOCUSED_RAG = [
    ("KGS FS551 저압 배관에서 30 kPa 이하이면 기밀시험을 생략할 수 있나요?", ["FS551"], "기밀시험·생략 조건"),
    ("KGS FS551 내압시험을 생략하거나 다른 시험으로 갈음할 수 있는 경우가 있나요?", ["FS551"], "내압시험·생략 조건"),
    ("KGS FS551 내압시험의 시험매체와 시험압력, 합격판정을 설명해줘.", ["FS551"], "내압시험 절차"),
    ("KGS FS551 수소 배관의 기밀시험과 내압시험은 어떻게 다른가요?", ["FS551"], "기밀·내압 비교"),
    ("KGS FS551 배관 용접부 비파괴검사에서 확인하는 항목을 알려줘.", ["FS551"], "용접부 검사"),
    ("KGS FU671 수소연료사용시설의 가스누출검지경보장치 설치 조건을 설명해줘.", ["FU671"], "누출검지경보장치"),
    ("KGS FP217 저장식 수소연료 충전시설의 저장탱크와 보호시설 이격거리 기준을 설명해줘.", ["FP217"], "이격거리"),
    ("KGS FP216 제조식 수소연료 충전시설의 적용범위와 FP217과의 차이를 비교해줘.", ["FP216", "FP217"], "적용범위 비교"),
    ("KGS FP111 고압가스 특정제조시설에서 수소 배관이 자동으로 포함되는지 판단 기준을 알려줘.", ["FP111"], "적용범위 경계"),
    ("KGS FS551 정기검사에서 기밀시험을 매번 해야 하는지 조건을 설명해줘.", ["FS551"], "정기검사·기밀시험"),
    ("고압가스 안전관리법에서 고압가스설비와 처리설비의 차이를 설명해줘.", [], "법정 정의"),
    ("수소경제 육성 및 수소안전관리에 관한 법률의 수소연료공급시설 정의와 적용 범위를 설명해줘.", [], "수소법 정의"),
    ("액화석유가스의 안전관리 및 사업법에서 저장탱크 검사와 배관 검사를 어떻게 구분하나요?", [], "액법 검사"),
    ("도시가스사업법상 가스공급시설의 안전관리 의무를 근거 조항과 함께 설명해줘.", [], "도법 안전관리"),
    ("사규 2201-1의 작업허가와 안전확인 절차를 순서대로 설명해줘.", ["2201-1"], "사규 작업허가"),
    ("사규 2400-1에서 검사 결과가 부적합일 때 후속 조치를 정리해줘.", ["2400-1"], "사규 부적합 조치"),
    ("사규 2302-1의 기록 보존과 보고 책임을 실무자 관점에서 설명해줘.", ["2302-1"], "사규 기록·보고"),
    ("KGS FS551에서 방호벽과 차량 진입 방지 조치가 같은 의미인지 구분해줘.", ["FS551"], "방호·차량진입"),
    ("KGS FS551의 기밀시험 생략 조건과 30 kPa 시험압력 예외가 같은 규정인지 비교해줘.", ["FS551"], "예외 비교"),
    ("KGS FS551에서 압력계로 기밀시험을 판정할 때 확인해야 할 조건을 설명해줘.", ["FS551"], "기밀시험 판정"),
]


LLM_GAP = [
    "밀폐공간에서 수소 관련 작업을 할 때 작업자가 지켜야 할 주의사항을 충분히 설명해줘.",
    "수소 누출이 의심될 때 일반적인 초기 대응 순서와 절대 하면 안 되는 행동을 알려줘.",
    "수소 작업자의 개인보호구를 위험별로 추천해줘.",
    "수소가 공기 중에서 퍼지는 특성과 작업자가 이해해야 할 위험을 쉽게 설명해줘.",
    "밀폐된 장소에서 수소 작업 전 환기와 가스 측정을 어떻게 계획해야 하나요?",
    "수소 설비 화재가 발생했을 때 일반적인 대피와 신고 원칙을 설명해줘.",
    "수소 배관의 수소취성 위험을 재료공학 관점에서 설명해줘.",
    "수소충전소 작업자가 교대 전에 확인할 일반 안전 체크리스트를 만들어줘.",
    "수소 누출 감지기가 오작동할 때 원인과 점검 방향을 설명해줘.",
    "수소 설비의 정전기와 점화원 위험을 일반적인 안전 원칙으로 설명해줘.",
    "수소 작업에서 산소농도와 가연성가스 농도를 왜 함께 확인해야 하는지 알려줘.",
    "수소 저장탱크 주변에 차량이 접근했을 때 일반적인 위험과 대응을 설명해줘.",
    "수소 배관을 처음 설계하는 사람이 놓치기 쉬운 안전 검토 항목을 정리해줘.",
    "수소 작업 중 압력이 갑자기 상승할 때 가능한 원인과 일반 대응을 알려줘.",
    "수소 설비 유지보수 작업 전 에너지 차단과 잔류가스 처리 원칙을 설명해줘.",
    "수소충전 작업 중 통신이 끊겼을 때 일반적으로 어떤 상태를 확인해야 하나요?",
    "수소와 질소를 혼동하면 안 되는 이유를 안전관리 관점에서 설명해줘.",
    "수소 작업장 환기 설계에서 자연환기와 강제환기의 차이를 알려줘.",
    "수소 설비에서 누출 냄새에 의존하면 안 되는 이유를 설명해줘.",
    "수소 안전교육을 처음 받는 직원에게 핵심 위험을 쉽게 설명해줘.",
] * 3


CLARIFICATION = [
    "그 기준에서 생략할 수 있어?", "수소 배관은 어떻게 해?", "내압시험 압력은?",
    "저장탱크 주변 거리는?", "그 장치 꼭 설치해야 해?", "검사 결과가 이상하면?",
    "이 기준의 예외는?", "몇 미터 띄워야 해?", "누가 책임져?", "이거 법에 걸려?",
    "그럼 지금은?", "뭐부터 확인해?", "저장설비 기준 알려줘", "수소법 적용돼?",
    "기밀시험 몇 분이야?", "압력밸브 역할은?", "충전기는 어디에 둬?", "안전거리는 얼마야?",
    "이 시설도 포함돼?", "검사해야 돼?", "이거 맞는 기준이야?", "어떤 법을 봐야 해?",
    "그 배관 기준 좀 찾아줘", "수소 관련 기준 정리해줘", "이 장치의 역할은?",
    "예외가 있나요?", "허가가 필요한가요?", "설치 위치는요?", "점검 주기는요?",
] * 3


OFF_TOPIC = [
    "오늘 서울 날씨 알려줘", "김치찌개 맛있게 끓이는 법", "파이썬으로 엑셀 파일 만드는 법",
    "이번 주말 부산 여행 코스 추천해줘", "고양이가 밤에 우는 이유가 뭐야", "재미있는 영화 추천해줘",
    "비트코인 가격 전망 알려줘", "영어로 자기소개 문장 만들어줘", "다이어트 식단을 짜줘",
    "가스비를 아끼는 생활 습관 알려줘", "가스레인지 청소하는 방법 알려줘", "수소차를 사도 괜찮을까?",
    "압력밥솥 사용법을 알려줘", "오늘 축구 경기 결과 알려줘", "엑셀에서 합계를 계산하는 방법",
    "친구에게 사과하는 문자를 써줘", "주식 차트 보는 방법 알려줘", "잠이 잘 오게 하는 방법",
    "고양이 이름을 지어줘", "주말에 볼 드라마 추천해줘",
] * 2


def _load_previous() -> list[dict]:
    path = ROOT / "eval" / "domain_queries.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["quality_category"] = "rag_direct"
        # A law-name hit is not enough to prove a duty, permit, penalty, or
        # appointment rule.  The correct result may be a disclosed LLM
        # fallback when the local law PDF returned only a definition article.
        row["expected_answer_mode"] = (
            ["rag", "llm_only"]
            if re.search(r"법|도법|액법|고법|수소법", str(row.get("query", "")))
            and re.search(r"의무|책임|허가|신고|벌칙|벌금|안전관리자|선임", str(row.get("query", "")))
            else ["rag"]
        )
        row["target_chars"] = 320
        rows.append(row)
    return rows


def build_cases() -> list[dict]:
    cases = _load_previous()
    sequence = len(cases) + 1
    for query, codes, note in FOCUSED_RAG:
        expected_modes = (
            ["rag", "llm_only"]
            if re.search(r"법|도법|액법|고법|수소법", query)
            and re.search(r"의무|책임|허가|신고|벌칙|벌금|안전관리자|선임", query)
            else ["rag"]
        )
        cases.append({
            "id": f"quality-rag-{sequence:04d}",
            "quality_category": "rag_direct",
            "query": query,
            "expected_doc_codes": codes,
            "expected_answer_mode": expected_modes,
            "target_chars": 420,
            "note": note,
            "source": "focused-grounding-bank",
        })
        sequence += 1
    for prefix, values, category, modes, target in (
        ("quality-gap", LLM_GAP, "llm_fallback", ["llm_only", "rag"], 420),
        ("quality-clarify", CLARIFICATION, "clarification", ["clarification"], 100),
        ("quality-offtopic", OFF_TOPIC, "off_topic", ["llm_only"], 180),
    ):
        for query in values:
            case_modes = modes
            case_target = target
            if category == "clarification" and not re.search(
                r"^그\s*(?:기준|장치)|^그럼|^뭐부터|^저장설비\s*기준|^수소법\s*적용|^어떤\s*법|^그\s*배관\s*기준|^수소\s*관련\s*기준",
                query,
            ):
                # A terse question that still names a concrete operation or
                # safety attribute should receive a qualified best-effort
                # answer, not a dead-end clarification request.
                case_modes = ["clarification", "rag", "llm_only"]
                case_target = 320
            cases.append({
                "id": f"{prefix}-{sequence:04d}",
                "quality_category": category,
                "query": query,
                "expected_doc_codes": [],
                "expected_answer_mode": case_modes,
                "target_chars": case_target,
                "note": category,
                "source": "focused-quality-bank",
            })
            sequence += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "quality_queries.jsonl")
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for case in cases:
        key = case["quality_category"]
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"total": len(cases), "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
