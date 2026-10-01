"""Build a reproducible robustness bank for noisy and out-of-domain questions.

The purpose of this bank is not to reward keyword matching.  It checks whether
the router can distinguish (1) genuinely underspecified requests, (2) clear
technical intent expressed with slang or profanity, (3) non-standard field
terms and typos, and (4) questions that do not belong to the safety corpus.
Each case carries the expected *handling* so the answer evaluator can detect a
bad route even when the generated prose happens to sound plausible.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(category: str, prefix: str, expected: str, values: list[str], note: str) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{index:03d}",
            "category": category,
            "query": query,
            "expected_behavior": expected,
            "expected_answer_mode": {
                "clarify": ["clarification"],
                "recover_rag": ["rag"],
                "llm_only": ["llm_only"],
                "clarify_or_llm_only": ["clarification", "llm_only"],
            }[expected],
            "note": note,
            "source": "handwritten-robustness-bank",
        }
        for index, query in enumerate(values, start=1)
    ]


def build_cases() -> list[dict]:
    cases: list[dict] = []

    # No document, facility, action, or legal question is identified.  The
    # correct response is a short, useful clarification rather than a random
    # hit from the corpus.
    cases += _rows(
        "ambiguous",
        "amb",
        "clarify",
        [
            "그거 기준 뭐야?", "이거 어떻게 해?", "저 기준 알려줘", "몇 미터 띄워?",
            "압력 얼마야?", "탱크는 어떻게 해?", "검사해야 돼?", "법에 걸려?",
            "그 장치 역할이 뭐야?", "이전 내용 기준으로 해줘", "뭘 설치해야 해?",
            "어디까지 적용돼?", "이건 안전해?", "몇 분 유지해?", "누가 담당이야?",
            "그거 허용돼?", "기준 좀 찾아봐", "뭐가 맞는 거야?", "이거 해도 돼?",
            "얼마나 떨어져야 해?", "어떤 검사를 해?", "문제 없지?", "이 상황은 어때?",
            "그 규정 적용돼?", "어느 법을 봐야 해?", "필요한 서류가 뭐야?", "언제 점검해?",
            "어떻게 막아?", "어디에 설치해?", "이거 신고해야 하나?", "대상인지 모르겠어",
            "무슨 기준으로 판단해?", "그럼 지금은?", "그 부분만 알려줘", "수치가 어떻게 돼?",
            "이게 맞아?", "뭐부터 보면 돼?", "기준이 몇 조야?", "그 시설은요?",
            "이런 경우는 어떻게 해?", "누가 확인해?", "왜 필요한데?", "기준이 뭐였지?",
            "몇 개를 둬야 해?", "어떤 장치가 필요해?", "검사 결과가 이상하면?", "이거 기준 찾아줘",
            "설치 위치 알려줘", "운영은 어떻게 해?", "이게 법적으로 괜찮아?", "전부 정리해줘",
        ],
        "식별자·시설·행위가 부족한 질문은 확인 질문으로 전환",
    )

    # These are intentionally somewhat vague but do contain a safety object.
    # They should still ask for the missing action/condition instead of picking
    # an arbitrary paragraph about that object.
    cases += _rows(
        "ambiguous",
        "ambtech",
        "clarify",
        [
            "저장탱크 주변은?", "수소 배관은?", "방호벽은 어떻게?", "충전소 거리는?",
            "가스가 새면?", "압력조정기는?", "안전밸브는 언제?", "탱크로리는요?",
            "기밀시험은 몇 분?", "내압시험 압력은?", "수소법 기준은?", "고법에서는?",
            "액법 적용은?", "도법 시설은?", "방폭구역은 어떻게 잡아?", "차량 진입은?"]
        + [
            "배관 검사 뭐가 있어?", "저장설비 기준 알려줘", "충전기 설치는?", "누출검지는?",
            "접지는 어디까지?", "환기는 어떻게?", "비파괴검사는?", "용접부는?",
            "압력계 교정은?", "안전관리자는?", "사고 나면?", "허가 절차는?",
            "변경 신고는?", "정기점검은?", "비상차단은?", "방류둑은?",
            "가스통 보관은?", "튜브트레일러는?", "액체수소는?", "연료전지는?"]
        + [
            "그 탱크의 거리는?", "이 배관의 압력은?", "그 검사 주기는?", "이 기준의 예외는?",
            "그 장치가 고장 나면?", "이 시설은 허가 대상?", "그 수치는 왜 그래?", "이 조항만 보면?",
            "저장탱크 옆에 뭘 둬?", "수소충전소 주변은?", "법령상 의무는?", "사규에 있어?",
            "현장에서는 뭘 확인해?", "설계할 때 주의점은?", "운전 중에는?", "정지할 때는?"]
        + [
            "그 기준에서 검사 항목은?", "탱크 주변 차량은?", "배관 매설 깊이는?", "압력 낮으면?",
            "누출 경보가 울리면?", "안전거리가 맞나?", "기준 문서가 뭐지?", "관련 조항 보여줘",
            "이 설비도 포함돼?", "어느 부서가 맡아?", "기록은 남겨?", "승인 받아야 해?",
            "현장 조치만 알려줘", "전체 절차는?", "기준 비교해줘", "적용 범위는?"]
        + [
            "배관이 긴 경우는?", "탱크가 여러 개면?", "실내 설치면?", "옥외 설치면?",
            "저압이면?", "고압이면?", "액화가스면?", "기체수소면?"]
        ,
        "대상은 있으나 조건·행위·문서가 부족한 질문은 필요한 범위를 되묻기",
    )

    # Mild profanity, chat shorthand, spacing errors, and colloquial wording;
    # the technical intent is sufficiently clear to normalize and retrieve.
    cases += _rows(
        "slang",
        "slang",
        "recover_rag",
        [
            "저장탱크 옆에 차 못 대게 하는 거 뭐임?", "가스 새면 뭐부터 해야됨?",
            "방호벽 이격거리 몇이냐", "과압밸브 그거 왜 달아?", "FS551 기밀시험 몇분임?",
            "수소충전소 위험구역 어떻게 잡음?", "압력조정기 점검주기 알려줘 ㅈㄴ 헷갈림",
            "법 조항 좀 찾아줘 개빡세네", "가스통 옆에 뭐 설치함?", "탱크로리 들어오면 어케 막음?",
            "누출감지기 꼭 달아야 됨?", "배관 용접 검사 뭐함?", "안전밸브 터지는 압력은?",
            "충전기 주변 몇 미터 띄움?", "저장탱크 과충전 막는 장치 뭐임?", "고압가스 검사 언제함?",
            "수소 배관 새면 어케 찾음?", "방폭 전기 뭐 써야됨?", "기밀시험 압력 얼마 넣음?",
            "이거 기준에 있냐? 좀 찾아봐", "가스 냄새 나는데 뭐부터 해야 돼?", "탱크 주변 차 세우면 안 됨?",
            "압력 떨어지면 어떤 조치함?", "안전관리자 누가 맡음?", "배관 묻을 때 깊이 얼마임?",
            "용접부 검사 빡세게 해야 하나?", "누출 나면 밸브부터 잠그는 거 맞음?", "수소법에 이거 나옴?",
            "고법에서 저장탱크 뭐라 함?", "액법 기준으로 가스통 보관 어케 함?", "도법 정압기 점검 뭐 봄?",
            "사규에 보고기한 있냐?", "이거 신고 안 하면 어찌 됨?", "방호벽 설치 위치 어디임?",
            "탱크 압력계 교정 언제 해?", "충전소 환기 어케 뚫어놔야 함?", "비상차단 버튼 어디 둠?",
            "가스 누설 경보기 오작동하면 어캄?", "배관 부식 검사 뭐로 함?", "내압시험 물로 해도 됨?",
            "안전거리 계산 좀 해줘", "저장설비 주차 막는 방법 뭐임?", "이 기준 개정됐냐?",
            "문서번호 FS 551 맞냐?", "KGS-FS551에서 검사 순서 뭐임?", "FS-551 적용범위 알려줘",
            "수소충전소에서 차단장치 꼭 있어야 함?", "기밀 유지시간 몇 분으로 봄?", "방폭구역 지정 기준 뭐임?",
            "과압 나면 안전밸브가 알아서 빼주는 거지?", "가스 설비 점검표 좀 뽑아줘", "이거 현장에서는 어떻게 처리함?",
        ],
        "비속어와 구어체를 반복하지 않고 정식 안전 용어로 바꿔 RAG",
    )

    # Colloquial/non-standard terms with a recoverable technical meaning.
    cases += _rows(
        "nonstandard",
        "term",
        "recover_rag",
        [
            "고법에서 가스통 검사 기준", "액법 저장통 이격거리", "도법 가스관 매설 기준",
            "수소법 충전기 안전관리", "압력밸브 역할과 설치기준", "가스 새는거 검사 방법",
            "탱크 안전거리 법적 기준", "방폭지역 구역 나누는 법", "가스통 보관 장소 기준",
            "충전기 주변 차량 거리", "저장통 과충전 방지", "파이프 기밀 검사", "가스 미터기 점검",
            "정압기 압력 세팅 기준", "릴리프 밸브 검사 주기", "가스 새는 센서 설치",
            "탱크차 진입 차단 기준", "배관 붙이는 용접 검사", "스텐 배관 사용 가능 여부",
            "수소관 재료 기준", "고압 탱크 검사 항목", "액체수소 저장통 기준", "기체수소 배관 기준",
            "가스 안전거리 몇 미터", "탱크 옆 방어벽 기준", "압력계 교정 방법", "가스 차단기 설치",
            "누설 체크 방법", "배관 새는지 확인", "저장시설 허가 기준", "충전소 신고 절차",
            "안전관리 담당자 자격", "비상벨 설치 위치", "가스 환풍기 기준", "탱크 기초 앵커 기준",
            "배관 땅속 깊이", "가스관 부식 방지", "용접 검사 엑스레이", "탱크 터짐 방지 장치",
            "수소차 충전 안전거리", "탱크로리 하역 절차", "가스통 교체 작업", "압력 낮추는 밸브",
            "위험구역 전기기계 기준", "배관 누출 알람", "저장탱크 주차 방지", "고압가스 법 기준",
            "사내 가스 작업 규칙", "검사 서류 뭐 내는지", "안전점검 체크리스트",
        ],
        "현장식 명칭을 문서의 표준 용어 후보로 확장해 검색",
    )

    # Clearly unrelated requests.  Even if a word such as '가스' appears in a
    # non-technical context, the router should not attach a KGS citation.
    cases += _rows(
        "off_topic",
        "off",
        "llm_only",
        [
            "오늘 서울 날씨 어때?", "이번 주말 부산 여행 코스 추천해줘", "김치찌개 레시피 알려줘",
            "고양이가 계속 우는데 이유가 뭐야?", "강아지 산책은 하루 몇 번이 좋아?", "오늘 축구 경기 결과 알려줘",
            "요즘 주식 뭐 사면 좋아?", "비트코인 가격 전망 알려줘", "재미있는 영화 추천해줘",
            "파이썬으로 계산기 만드는 법", "자바스크립트 오류 고쳐줘", "엑셀에서 합계 구하는 법",
            "이메일을 영어로 번역해줘", "짧은 시 한 편 써줘", "생일 축하 문구 만들어줘",
            "내일 비 오면 우산 챙겨야 해?", "맛집 추천해줘", "운동 루틴 짜줘", "다이어트 식단 알려줘",
            "집에서 커피 맛있게 내리는 법", "고양이 이름 지어줘", "강아지 사료 추천", "주말에 뭐 하지?",
            "넷플릭스 볼 만한 것", "한국 드라마 추천", "영어 회화 연습하자", "면접 자기소개 써줘",
            "보고서 문장을 예쁘게 다듬어줘", "파이썬 게임 하나 만들어줘", "HTML 버튼 색 바꾸기",
            "컴퓨터가 느린데 어떻게 해?", "윈도우 비밀번호 잊어버렸어", "사진 배경 지워줘",
            "노래 가사 찾아줘", "로또 번호 추천해줘", "주식 차트 읽는 법", "환율이 왜 오르지?",
            "가스비 아끼는 생활 팁", "가스레인지 청소 방법", "가스 요금 계산해줘", "탄산가스 음료 추천",
            "수소차 살 만해?", "수소의 원자번호가 뭐야?", "압력밥솥 사용법", "탱크 게임 공략 알려줘",
            "배관공 월급이 얼마야?", "안전한 비밀번호 만들어줘", "오늘 뉴스 요약해줘", "정치 뉴스 어떻게 생각해?",
            "여행용 가방 추천", "냉장고 정리 방법", "집에서 빵 굽는 법", "코딩 공부 순서 알려줘",
            "친구에게 사과하는 문자 써줘", "회의 일정 정리해줘", "내 이름으로 삼행시", "잠이 안 올 때 방법",
            "사진 속 글자 읽어줘", "음악 플레이리스트 추천", "한국어 맞춤법 검사해줘", "재택근무 집중 방법",
        ],
        "문서 검색·인용 없이 LLM 자체 판단임을 표시",
    )

    # A few field terms are genuinely ambiguous rather than merely informal.
    # Keep them in the same non-standard bucket, but require a clarification so
    # the evaluator does not reward an arbitrary choice between a safety valve
    # and a pressure regulator.
    for case in cases:
        if case["query"] == "압력밸브 역할과 설치기준":
            case["expected_behavior"] = "clarify"
            case["expected_answer_mode"] = ["clarification"]
            case["note"] = "압력밸브가 안전밸브인지 압력조정기인지 먼저 확인"

    # Some entries were written in a colloquial, code-less form but still ask
    # a concrete, searchable thing (distance, pressure, inspection, or role).
    # Those are valid recovery tests rather than clarification tests.
    concrete = re.compile(
        r"거리|이격|압력|수치|몇|주기|시간|분|절차|단계|순서|방법|역할|기능|설치|"
        r"검사|점검|시험|허가|신고|보고|서류|책임|의무|조항|누출|누설|차단|방지|"
        r"교정|매설|새면|샘|왜\s*(?:달아|설치|필요)",
        re.IGNORECASE,
    )
    specific = re.compile(
        r"KGS|가스|수소|배관|탱크|저장|충전|밸브|방호벽|방폭|용접|기밀|내압|"
        r"누출|누설|검지|도법|액법|고법|수소법|법령|사규|내규",
        re.IGNORECASE,
    )
    for case in cases:
        if case["category"] == "ambiguous" and specific.search(case["query"]) and concrete.search(case["query"]):
            case["expected_behavior"] = "recover_rag"
            case["expected_answer_mode"] = ["rag"]
            case["note"] = "문서번호는 없지만 질문 대상과 확인 항목이 구체적이므로 정식 용어로 복원 후 RAG"

    for case in cases:
        if case["query"] in {"가스 새면 뭐부터 해야됨?", "방호벽 이격거리 몇이냐"}:
            case["expected_behavior"] = "clarify"
            case["expected_answer_mode"] = ["clarification"]
            case["note"] = "시설·기준에 따라 조치나 거리가 달라지는 질문은 임의의 수치를 선택하지 않고 범위를 확인"

    # Keep the order stable: category blocks make manual review and resuming a
    # long API run predictable.
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "robustness_queries.jsonl")
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    print(json.dumps({"total": len(cases), "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
