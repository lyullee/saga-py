import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from saga.database import Database
from saga.rag import (
    CHAT_ONLY_NOTICE,
    LLM_LIMITED_NOTICE,
    LLM_ONLY_NOTICE,
    RagPipeline,
    _add_scope_boundary_note,
    _clean_extracted_text,
    _format_answer_for_display,
    _guard_operation_scope,
    _mark_llm_only_answer,
    _needs_query_clarification,
    _normalize_repeated_ordered_sections,
    _normalize_broken_ordered_items,
    _normalize_broken_emphasis_sections,
    _rag_answer_has_grounding_gaps,
    _query_evidence_is_misaligned,
    _remove_unverified_measurements,
    _specific_inspection_heading,
    _uses_fast_grounding_model,
)
from saga.text import extract_document_codes
from saga.schemas import AnswerReview, ChatRequest, ChatResponse, ChatTurn, Citation, EvidenceClaim, GroundedClaims


class ReasonerMustNotRun:
    async def structured(self, *_args, **_kwargs):
        raise AssertionError("The LLM must not run for an absent explicit document code")


def test_specific_welding_inspection_maps_to_joining_clause_not_whole_overview():
    assert _specific_inspection_heading("KGS FS551 배관 용접부 검사 절차") == "접합"
    assert _specific_inspection_heading("KGS FS551 PE배관 융착 검사 절차") == "접합"
    assert _specific_inspection_heading("KGS FS551의 전체 검사 절차") is None
    assert _specific_inspection_heading("KGS FS551 정기검사 절차") == "정기검사"
    assert _specific_inspection_heading(
        "KGS FS551 공사 전, 시공 중, 정기검사 단계별 체크리스트"
    ) is None


def test_simple_facts_use_fast_grounding_but_reasoning_uses_primary_model():
    assert _uses_fast_grounding_model("KGS FS551 기밀시험 압력 기준은?", "fact")
    assert not _uses_fast_grounding_model("왜 이 압력 예외가 저압 배관에는 적용되지 않나요?", "fact")
    assert _uses_fast_grounding_model(
        "KGS FS551 시공감리와 정기검사 때의 기밀시험 구분을 근거 조항으로 요약해줘.",
        "summary",
    )
    assert not _uses_fast_grounding_model(
        "수소 배관망 전체의 위험도를 분석하고 대응 시나리오를 평가해줘.", "summary"
    )
    assert not _uses_fast_grounding_model("KGS FS551 검사 절차를 단계별로 요약해줘", "procedure")


def test_llm_only_notice_is_idempotent_and_explicit():
    assert _mark_llm_only_answer("일반적인 답변입니다.").startswith(LLM_ONLY_NOTICE)
    assert _mark_llm_only_answer(LLM_ONLY_NOTICE + "\n\n이미 표시됨") == LLM_ONLY_NOTICE + "\n\n이미 표시됨"


def test_role_answer_does_not_turn_adjacent_test_requirements_into_a_device_purpose():
    related_only = [{
        "content": "과압안전장치는 압력 및 온도에 견디는 구조와 재질로 하고 작동시험을 한다.",
    }]
    direct_role = [{
        "content": "과압이 발생하면 압력을 방출하여 설비를 보호한다.",
    }]

    assert not RagPipeline._has_direct_role_evidence(related_only)
    assert RagPipeline._has_direct_role_evidence(direct_role)


def test_dense_grounded_answer_gets_readable_markdown_sections_without_rewriting_text():
    answer = (
        "핵심은 출입을 통제하는 것입니다. 차량 진입금지 표지를 설치합니다 [1] "
        "물리적 차단장치를 함께 배치합니다 [1] 정기적으로 상태를 확인합니다 [1] "
        "손상이나 무단 이동이 발견되면 즉시 보수하고 관리 책임자를 정해 기록을 남깁니다 [1]"
    )

    formatted = _format_answer_for_display(answer)

    assert "### 핵심 답변" in formatted
    assert "### 구체적으로 보면" in formatted
    assert "차량 진입금지 표지를 설치합니다 [1]" in formatted
    assert "물리적 차단장치를 함께 배치합니다 [1]" in formatted


def test_repeated_numbered_section_labels_are_renumbered_but_steps_are_preserved():
    answer = (
        "1. 설치 위치\n\n"
        "- 배관 끝에 설치합니다. [1]\n"
        "- 접근 가능한 곳에 둡니다. [1]\n\n"
        "1. 작동·검사\n\n"
        "- 시험으로 확인합니다. [2]\n\n"
        "1. 수격 방지\n\n"
        "- 역류를 막습니다. [3]\n\n"
        "1. 1단계에서 밸브를 잠급니다. [4]\n"
        "2. 2단계에서 압력을 확인합니다. [4]"
    )
    normalized = _normalize_repeated_ordered_sections(answer)
    assert "1. 설치 위치" in normalized
    assert "2. 작동·검사" in normalized
    assert "3. 수격 방지" in normalized
    assert "1. 1단계에서 밸브를 잠급니다. [4]" in normalized
    assert "2. 2단계에서 압력을 확인합니다. [4]" in normalized


def test_standalone_markdown_number_is_joined_to_following_title():
    answer = (
        "현장에서 확인할 문서\n\n"
        "1.\n\n"
        "**가스기술기준위원회가 정한 상세기준** – 설계·제조·검사 기준입니다.\n\n"
        "2.\n\n"
        "**안전관리자 선임·해임 기록** – 관련 기록을 확인합니다."
    )

    normalized = _normalize_broken_ordered_items(answer)

    assert "1. **가스기술기준위원회가 정한 상세기준**" in normalized
    assert "2. **안전관리자 선임·해임 기록**" in normalized
    assert "\n1.\n" not in normalized


def test_split_bold_numbered_heading_becomes_markdown_heading():
    answer = "**1.\n\n표지판의 내용과 부착 위치**\n\n- 시공 정보를 표시합니다."

    normalized = _normalize_broken_emphasis_sections(answer)

    assert normalized.startswith("### 1. 표지판의 내용과 부착 위치")
    assert "**1." not in normalized
    assert normalized.count("표지판의 내용과 부착 위치") == 1


def test_composite_vessel_introduction_is_searchable_without_clarification():
    assert not _needs_query_clarification("복합가스 용기 기준 소개해줘", [])


@pytest.mark.asyncio
async def test_related_standard_followup_reuses_technical_history(tmp_path: Path):
    class FollowupReasoner:
        async def structured(self, _prompt, schema, _name):
            return schema(
                rewritten_query="복합가스 용기 관련 기준",
                intent="general",
                domain="GENERAL",
                keywords=["복합가스", "용기"],
                document_codes=[],
                retrieval_required=False,
            )

        async def answer(self, _prompt, **_kwargs):
            return "관련 기준을 확인할 수 있도록 일반적인 범위를 설명하겠습니다."

    database = Database(tmp_path / "followup.db")
    database.initialize()
    settings = SimpleNamespace(
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        retrieval_limit=32,
        context_limit=24,
        max_context_chars=42000,
        hybrid_search_enabled=True,
        answer_review_enabled=False,
        reasoning_effort="low",
    )
    response = await RagPipeline(settings, database, FollowupReasoner()).run(
        ChatRequest(
            message="관련 기준은 없어?",
            history=[
                ChatTurn(role="user", content="복합가스 용기 기준 소개해줘"),
                ChatTurn(role="assistant", content="앞서 확인한 내용입니다."),
            ],
        )
    )

    assert response.intent != "clarification"


def test_operation_scope_guard_does_not_turn_transfer_ban_into_vehicle_entry_ban():
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="RULE", doc_code="2201-1",
        title="LPG시설 지침", hierarchy="제5-3조", page=20,
        filename="2201-1.pdf", excerpt="저장탱크에서 탱크로리로의 이입작업을 금지하도록 한다.", score=1.0,
    )
    response = ChatResponse(
        conversation_id="c", log_id=1, answer="차량 진입을 금지합니다 [1]",
        citations=[citation], rewritten_query="", intent="fact", model="test",
    )

    guarded = _guard_operation_scope(
        ChatRequest(message="저장탱크 주변 차량 진입 방지 대책"), response, response.answer
    )

    assert "이입작업 금지" in guarded
    assert "차량의 물리적 진입 자체를 금지" in guarded
    assert "차단봉" in guarded


def test_numbered_standard_list_repairs_ocr_and_removes_amendment_metadata():
    content = (
        "검사 항목은 다음과 같다. (1) 2.4.4 가스설비 확인 <개정 20.9.4> "
        "(2) 2.5.9.1 기밀성능의 학인 및 파이 프덕트 시설 의 확인. "
        "(3) 2.10.1 ᄃ자 형태의 표지 확인"
    )

    intro, items = RagPipeline._numbered_source_items(content)

    assert intro == "검사 항목은 다음과 같다."
    assert len(items) == 3
    assert "<개정" not in items[0][1]
    assert "기밀성능의 확인" in items[1][1]
    assert "파이프덕트 시설의 확인" in items[1][1]
    assert "ㄷ자 형태" in items[2][1]


def test_pdf_line_wrap_repair_joins_known_split_words_without_rewriting_sentences():
    source = "가스사용시 설로서 가스누출자동차단 기를 설치한 다."

    assert _clean_extracted_text(source) == "가스사용시설로서 가스누출자동차단기를 설치한다."
    assert _clean_extracted_text("일반도시가 스사업의") == "일반도시가스사업의"
    assert _clean_extracted_text("경보를 울리는 것으로 한 다.") == "경보를 울리는 것으로 한다."
    assert _clean_extracted_text("압력이상으로 실시하지 않 을수 있다.") == "압력 이상으로 실시하지 않을 수 있다."
    assert _clean_extracted_text("30kPa 이하 인 것은 시험압력으로 할수 있다.") == "30kPa 이하인 것은 시험압력으로 할 수 있다."
    assert _clean_extracted_text("차량 및그 밖의 작업") == "차량 및 그 밖의 작업"
    assert _clean_extracted_text("다 음에 해당하는 경우") == "다음에 해당하는 경우"
    assert _clean_extracted_text("현장 에서 비 파괴시험을 실시한다. 배관에 대해 서는") == "현장에서 비파괴시험을 실시한다. 배관에 대해서는"
    assert _clean_extracted_text("별도의 비파 괴시험을 하지 않는다.") == "별도의 비파괴시험을 하지 않는다."
    assert _clean_extracted_text("천정내부.바닥.벽속에 불연성 재료 로 덮는다.") == "천정 내부·바닥·벽속에 불연성 재료로 덮는다."
    assert _clean_extracted_text("고 압가스 제조.압축시설의 시설.기술.검사 기준") == "고압가스 제조·압축시설의 시설·기술·검사 기준"
    assert _clean_extracted_text("가스사용시설의 설 치·운영 및 검사에 적용한다.") == "가스사용시설의 설치·운영 및 검사에 적용한다."
    assert _clean_extracted_text("기밀시 험은 최고 사용압력 이상으로 한다.") == "기밀시험은 최고사용압력 이상으로 한다."
    assert _clean_extracted_text("폭발 하한계의 1/4 이하 이하로 한다.") == "폭발 하한계의 1/4 이하로 한다."

    async def answer(self, *_args, **_kwargs):
        raise AssertionError("The LLM must not run for an absent explicit document code")


@pytest.mark.asyncio
async def test_missing_explicit_code_returns_clarification_without_llm(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    for code in ("FP551", "FS551", "FU551"):
        database.replace_document(
            {
                "doc_type": "CODE", "doc_code": code, "title": f"{code} 기준",
                "filename": f"{code}.pdf", "file_path": str(tmp_path / f"{code}.pdf"),
                "file_hash": code, "page_count": 1,
            },
            [],
        )
    pipeline = RagPipeline(SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun())

    response = await pipeline.run(ChatRequest(message="KGS FF551은 뭐야?"))

    assert response.intent == "clarification"
    assert response.citations == []
    assert [item.doc_code for item in response.suggestions] == ["FP551", "FS551", "FU551"]
    assert "무관한 문서를 근거로 대신 답하지 않았습니다" in response.answer


@pytest.mark.asyncio
async def test_meaningful_standalone_question_gets_general_llm_answer_instead_of_dead_end(tmp_path: Path):
    class GeneralReasoner:
        async def answer(self, _prompt, **_kwargs):
            return "반복 고장은 동일 설비와 증상의 재발 패턴을 묶어 빈도와 원인을 비교하는 방식으로 분석합니다."

    database = Database(tmp_path / "standalone.db")
    database.initialize()
    response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="test-model",
            answer_review_enabled=False,
            reasoning_effort="low",
        ),
        database,
        GeneralReasoner(),
    ).run(ChatRequest(message="점검 데이터로 반복 고장을 찾는 방법을 알려줘"))

    assert response.answer_mode == "llm_only"
    assert response.intent == "general"
    assert "현재 질문만으로는" not in response.answer
    assert LLM_LIMITED_NOTICE in response.answer
    assert "반복 고장" in response.answer


@pytest.mark.asyncio
async def test_non_rag_question_uses_llm_with_explicit_disclosure(tmp_path: Path):
    class GeneralReasoner:
        def __init__(self):
            self.answer_prompt = ""

        async def structured(self, _prompt, schema, _name):
            return schema(
                rewritten_query="오늘 점심 메뉴 추천",
                intent="fact",
                domain="CROSS",
                keywords=["점심", "메뉴", "추천"],
                document_codes=[],
                retrieval_required=True,
            )

        async def answer(self, prompt, **_kwargs):
            self.answer_prompt = prompt
            return "가벼운 메뉴를 원한다면 비빔밥이나 샐러드를 추천합니다."

    database = Database(tmp_path / "general.db")
    database.initialize()
    reasoner = GeneralReasoner()
    settings = SimpleNamespace(
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        retrieval_limit=32,
        context_limit=24,
        max_context_chars=42000,
        hybrid_search_enabled=True,
    )
    response = await RagPipeline(settings, database, reasoner).run(
        ChatRequest(message="오늘 점심 메뉴 추천해줘")
    )

    assert response.answer_mode == "llm_only"
    assert response.citations == []
    assert response.answer.startswith(LLM_ONLY_NOTICE)
    assert LLM_ONLY_NOTICE in reasoner.answer_prompt


@pytest.mark.asyncio
async def test_explicit_chat_mode_skips_planner_and_retrieval(tmp_path: Path):
    class ChatOnlyReasoner:
        def __init__(self):
            self.answer_prompt = ""

        async def structured(self, *_args, **_kwargs):
            raise AssertionError("ordinary chat must not invoke the RAG planner")

        async def answer(self, prompt, **_kwargs):
            self.answer_prompt = prompt
            return "오늘은 부담 없이 우선순위를 정해 보시면 좋겠습니다."

    database = Database(tmp_path / "chat-only.db")
    database.initialize()
    reasoner = ChatOnlyReasoner()
    settings = SimpleNamespace(
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        answer_review_enabled=False,
        reasoning_effort="low",
    )

    response = await RagPipeline(settings, database, reasoner).run(
        ChatRequest(message="오늘 하루를 어떻게 시작하면 좋을까?", mode="chat")
    )

    assert response.mode == "chat"
    assert response.answer_mode == "llm_only"
    assert response.citations == []
    assert response.answer.startswith(CHAT_ONLY_NOTICE)
    assert "문서 검색 없이" in reasoner.answer_prompt


@pytest.mark.asyncio
async def test_answer_review_revises_draft_and_persists_it(tmp_path: Path):
    class ReviewReasoner:
        async def structured(self, _prompt, schema, _name):
            return schema(
                rewritten_query="점심 추천",
                intent="general",
                domain="GENERAL",
                keywords=["점심"],
                document_codes=[],
                retrieval_required=False,
            )

        async def answer(self, _prompt, **_kwargs):
            return "아무거나 드세요."

        async def structured_model(self, _prompt, schema, name, *_args):
            assert name == "answer_review"
            assert schema is not None
            return AnswerReview(approved=False, revised_answer="가볍게 드시려면 비빔밥을 추천합니다.")

    database = Database(tmp_path / "review.db")
    database.initialize()
    settings = SimpleNamespace(
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        retrieval_limit=32,
        context_limit=24,
        max_context_chars=42000,
        hybrid_search_enabled=True,
        answer_review_enabled=True,
        answer_review_model="gpt-oss-20b",
        reasoning_effort="low",
    )
    response = await RagPipeline(settings, database, ReviewReasoner()).run(
        ChatRequest(message="점심 추천해줘")
    )

    assert response.answer.startswith(LLM_ONLY_NOTICE)
    assert "비빔밥" in response.answer
    with database.connect() as connection:
        row = connection.execute("SELECT answer, model FROM chat_logs WHERE id = ?", (response.log_id,)).fetchone()
    assert "비빔밥" in row["answer"]
    assert row["model"].endswith("+review")


@pytest.mark.asyncio
async def test_best_effort_grounding_replaces_refusal_with_qualified_source_answer():
    class RefusingReasoner:
        async def answer(self, _prompt, **_kwargs):
            return "관련 근거는 찾았지만 답변을 보류합니다."

    citation = Citation(
        number=1,
        document_id=1,
        chunk_id=1,
        doc_type="CODE",
        doc_code="FU671",
        title="수소자동차 충전소 기준",
        hierarchy="2.1.2 보호시설과의 거리",
        page=12,
        filename="FU671.pdf",
        excerpt="저장탱크와 보호시설 사이에는 기준에서 정한 거리를 확보한다.",
        score=1.0,
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_fast_model="gpt-oss-20b"),
        None,
        RefusingReasoner(),
    )

    answer, citations = await pipeline._best_effort_grounded_answer(
        ChatRequest(message="저장탱크와 보호시설 사이 이격거리는?"),
        [citation],
        [{"content": citation.excerpt}],
    )

    assert "답변을 보류" not in answer
    assert "자동 검증" in answer
    assert "FU671 기준에서 확인되는 내용은" in answer
    assert citations == [citation]


def test_broad_confined_space_hydrogen_question_is_retrievable_not_clarification():
    query = "밀폐공간에서 수소 관련 작업을 할 때 주의할 점은 무엇인가요?"
    assert _needs_query_clarification(query, []) is False


def test_full_law_name_is_retrievable_without_clarification():
    assert _needs_query_clarification("도시가스사업법의 목적과 적용 대상은 무엇인가요?", []) is False


def test_llm_fallback_drops_unverified_numeric_thresholds():
    answer = (
        f"{LLM_LIMITED_NOTICE}\n\n"
        "일반적으로 작업 전에는 누출 여부를 확인해야 하며, 농도가 10% LEL에 도달하면 즉시 대피합니다."
    )
    cleaned = _remove_unverified_measurements(answer, "밀폐공간에서 수소 관련 작업 주의사항")
    assert "10% LEL" not in cleaned
    assert "구체적인 수치와 판단 기준" in cleaned


def test_rag_review_guard_rejects_uncited_rewrite():
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="LAW", doc_code="LAW-X",
        title="테스트 법령", hierarchy="제1조(목적)", page=1,
        filename="LAW-X.pdf", excerpt="이 법은 가스시설의 안전을 확보하는 것을 목적으로 한다.", score=1.0,
    )
    assert _rag_answer_has_grounding_gaps("가스시설의 안전을 확보합니다 [1]", [citation]) is False
    assert _rag_answer_has_grounding_gaps("사업자는 매년 점검하고 제29조 의무를 이행해야 합니다 [1]", [citation]) is True


def test_misaligned_domain_hit_falls_back_instead_of_answering_another_topic():
    assert _query_evidence_is_misaligned(
        "밀폐공간에서 수소 작업을 할 때 주의사항",
        [{"content": "수소연료사용시설의 경계표지를 외부에 게시한다."}],
    ) is True
    assert _query_evidence_is_misaligned(
        "밀폐공간에서 수소 작업을 할 때 주의사항",
        [{"content": "작업장소 산소농도 측정과 환기 상태를 확인한다."}],
    ) is False


@pytest.mark.asyncio
async def test_best_effort_can_answer_from_llm_when_rag_cannot_prove_claim():
    class HelpfulReasoner:
        async def answer(self, prompt, **_kwargs):
            return "밀폐공간 작업 전에는 산소와 가연성 가스를 측정하고 환기·출입통제를 준비해야 합니다."

    pipeline = RagPipeline(
        SimpleNamespace(
            service_hub_fast_model="gpt-oss-20b",
            service_hub_model="gpt-oss-120b",
            reasoning_effort="low",
        ),
        None,
        HelpfulReasoner(),
    )
    citation = Citation(
        number=1,
        document_id=1,
        chunk_id=1,
        doc_type="CODE",
        doc_code="FU671",
        title="수소충전소 기준",
        hierarchy="2.1 안전관리",
        page=12,
        filename="FU671.pdf",
        excerpt="저장탱크의 재질과 시험 요건을 정한다.",
        score=1.0,
    )

    answer, citations = await pipeline._best_effort_grounded_answer(
        ChatRequest(message="밀폐공간에서 수소 관련 작업을 할 때 주의할 점은 무엇인가요?"),
        [citation],
        [{"content": citation.excerpt}],
        allow_llm_fallback=True,
    )

    assert answer.startswith(LLM_LIMITED_NOTICE)
    assert "밀폐공간 작업 전" in answer
    assert citations == []


@pytest.mark.asyncio
async def test_explicit_low_pressure_scope_limit_returns_short_source_cited_answer(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "AA012",
            "title": "가스시설 기준",
            "filename": "AA012.pdf",
            "file_path": str(tmp_path / "AA012.pdf"),
            "file_hash": "aa012-test",
            "page_count": 9,
        },
        [
            {
                "hierarchy": "[AA012] 1 일반사항 > 1.1 적용범위",
                "page": 9,
                "content": "이 기준은 저압 (3.3 kPa 이하) 전용 가스시설에 적용한다.",
                "search_text": "AA012 저압 3.3 kPa 이하 전용 가스시설",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(message="KGS AA012에서 저압 전용으로 제한하는 압력 상한은 얼마야?")
    )

    assert response.intent == "fact"
    assert response.answer == (
        "AA012 적용범위의 압력 상한은 3.3 kPa입니다. "
        "원문은 ‘저압 (3.3 kPa 이하) 전용’으로 규정합니다. [1]"
    )
    assert len(response.citations) == 1
    assert response.citations[0].doc_code == "AA012"
    assert response.citations[0].page == 9
    assert response.citations[0].excerpt == "저압 (3.3 kPa 이하) 전용"


@pytest.mark.asyncio
async def test_explicit_single_code_scope_question_quotes_its_11_clause(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FP111",
            "title": "고압가스 특정제조 기준",
            "filename": "FP111.pdf",
            "file_path": str(tmp_path / "FP111.pdf"),
            "file_hash": "fp111-test",
            "page_count": 20,
        },
        [
            {
                "hierarchy": "[FP111] 1 일반사항 > 1.1 적용범위",
                "page": 13,
                "content": (
                    "이 기준은 「고압가스 안전관리법 시행령」(이하 ‘영’이라 한다) "
                    "제3조제1항제1호에 따른 고압가스 특정제조의 시설·기술·검사·감리·정밀안전검진에 적용한다."
                ),
                "search_text": "FP111 고압가스 특정제조 적용범위 시설 기술 검사 감리",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FP111의 적용범위와 고압가스 특정제조시설 중 수소 배관에 "
                "적용되는 범위를 간단히 설명해줘."
            )
        )
    )

    assert response.intent == "fact"
    assert response.citations and response.citations[0].doc_code == "FP111"
    assert response.citations[0].page == 13
    assert "FP111 1.1 적용범위" in response.answer
    assert "고압가스 특정제조의 시설·기술·검사·감리·정밀안전검진" in response.answer
    assert "모든 수소 배관에 자동 적용" in response.answer


def test_database_scope_search_accepts_compact_11_hierarchy(tmp_path: Path):
    database = Database(tmp_path / "scope.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "배관 기준",
            "filename": "FS551.pdf",
            "file_path": str(tmp_path / "FS551.pdf"),
            "file_hash": "fs551-scope",
            "page_count": 12,
        },
        [
            {
                "hierarchy": "[FS551] 1.1 적용범위",
                "page": 12,
                "content": "이 기준은 가스공급시설 중 가스배관에 적용한다.",
                "search_text": "FS551 1.1 적용범위 가스공급시설 가스배관",
            }
        ],
    )

    rows = database.search_document_scopes(["FS551"])

    assert len(rows) == 1
    assert rows[0]["doc_code"] == "FS551"


@pytest.mark.asyncio
async def test_fu671_facility_scope_is_not_conflated_with_hydrogen_gas_equipment(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-test",
            "page_count": 15,
        },
        [
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.1 적용범위",
                "page": 11,
                "content": (
                    "이 기준은 수소경제 육성 및 수소 안전관리에 관한 법률 제2조제9호에 따른 "
                    "수소연료사용시설의 시설·기술·검사에 대하여 적용한다."
                ),
                "search_text": "FU671 수소연료사용시설 법률 제2조제9호 적용범위",
            },
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의",
                "page": 11,
                "content": (
                    "이 기준에서 사용하는 용어의 뜻은 다음과 같다. "
                    "1.3.3 수소가스설비란 수소제조설비, 수소저장설비 및 연료전지와 "
                    "이들 설비를 연결하는 배관 및 그 부속설비 중 수소가 통하는 부분을 말한다."
                ),
                "search_text": "FU671 수소가스설비 수소제조설비 수소저장설비 연료전지",
            },
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "그럼 FU671 자체에서 수소연료사용시설에 포함되는 세부 설비를 정의해? "
                "정의가 없다면 1.1 조항이 법 제2조제9호를 인용한다는 점까지만 말해줘."
            )
        )
    )

    assert "용어정의에는 ‘수소연료사용시설’ 항목이 없습니다" in response.answer
    assert "제2조제9호" in response.answer
    assert "수소가스설비" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].hierarchy.endswith("1.1 적용범위")
    assert response.citations[0].page == 11


@pytest.mark.asyncio
async def test_single_code_scope_question_with_spaces_uses_exact_11_clause(tmp_path: Path):
    database = Database(tmp_path / "spaced-scope.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-spaced-scope",
            "page_count": 15,
        },
        [
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.1 적용범위",
                "page": 11,
                "content": (
                    "이 기준은 수소경제 육성 및 수소 안전관리에 관한 법률 제2조제9호에 따른 "
                    "수소연료사용시설의 시설·기술·검사에 대하여 적용한다."
                ),
                "search_text": "FU671 적용범위 대상 시설 수소연료사용시설",
            }
        ],
    )
    response = await RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    ).run(ChatRequest(message="KGS FU671의 적용 범위와 대상 시설을 알려줘."))

    assert response.intent == "fact"
    assert response.citations[0].doc_code == "FU671"
    assert response.citations[0].page == 11
    assert "FU671 1.1 적용범위" in response.answer
    assert "제2조제9호" in response.answer
    assert "수소연료사용시설의 시설·기술·검사" in response.answer


def test_fu671_scope_and_hydrogen_gas_equipment_definition_are_separate():
    scope = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1,
        "hierarchy": "[FU671] 1 일반사항 > 1.1 적용범위",
        "page": 11,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": "이 기준은 법 제2조제9호에 따른 수소연료사용시설의 시설·기술·검사에 적용한다.",
    }
    definition = {
        **scope,
        "chunk_id": 2,
        "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의",
        "content": "1.3.3 수소가스설비란 수소제조설비, 수소저장설비 및 연료전지와 이들 설비를 연결하는 배관 중 수소가 통하는 부분을 말한다.",
    }

    result = RagPipeline._fu671_scope_vs_equipment_definition(
        "FU671 적용범위가 수소가스설비 정의와 같은 의미인지 1.1과 1.3을 구분해줘",
        [scope],
        [definition],
    )

    assert result is not None
    rows, answer = result
    assert [row["chunk_id"] for row in rows] == [1, 2]
    assert "같은 의미로 볼 수 없습니다" in answer


@pytest.mark.asyncio
async def test_fu671_storage_tank_building_clearance_uses_protected_facility_table(tmp_path: Path):
    database = Database(tmp_path / "fu671-clearance.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설의 시설·기술·검사 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-clearance",
            "page_count": 100,
        },
        [
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의",
                "page": 11,
                "content": "1.3.2 ‘수소저장설비’란 수소를 충전·저장하기 위하여 지상 또는 지하에 고정 설치하는 저장탱크를 말한다.",
                "search_text": "FU671 1.3.2 수소저장설비 수소 저장탱크 지상 지하 고정",
            },
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의 > 1.3.6.1 제1종보호시설",
                "page": 12,
                "content": "1.3.6.1 제1종보호시설 (1) 학교 (2) 병원급 의료기관",
                "search_text": "FU671 1.3.6.1 제1종보호시설 학교 병원급 의료기관 건축물",
            },
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의 > 1.3.6.2 제2종보호시설",
                "page": 12,
                "content": "1.3.6.2 제2종보호시설 (1) 단독주택 및 공동주택 (2) 연면적 100m² 이상 1천m² 미만 건축물",
                "search_text": "FU671 1.3.6.2 제2종보호시설 단독주택 공동주택 건축물",
            },
            {
                "hierarchy": "[FU671] 2 시설기준 > 2.1 배치기준 > 2.1.1 화기와의 거리",
                "page": 14,
                "content": "2.1.1.5 화기를 사용하는 장소가 불연성 건축물 내에 있는 경우 수소저장설비로부터 수평거리 8m 이내의 그 건축물의 개구부는 방화문으로 폐쇄한다.",
                "search_text": "FU671 2.1.1 화기와의 거리 수소저장설비 불연성 건축물 개구부 8m",
            },
            {
                "hierarchy": "[FU671] 2 시설기준 > 2.1 배치기준 > 2.1.2 보호시설과의 거리",
                "page": 15,
                "content": "2.1.2 수소저장설비가 외면으로부터 보호시설까지 유지하여야 할 거리는 표 2.1.2에서 정한 거리 이상으로 한다. 사업소 안 및 전용공업지역 안의 보호시설은 제외한다.",
                "search_text": "FU671 2.1.2 보호시설과의 거리 수소저장설비 건축물 안전거리",
            },
            {
                "hierarchy": "[FU671] 2 시설기준 > 2.2 기초기준 > 2.2.1 지반조사",
                "page": 15,
                "content": "표 2.1.2 보호시설의 안전거리 저장능력(단위: m3) 제1종 보호시설 제2종 보호시설 1만 이하 17 12 1만 초과 2만 이하 21 14 2만 초과 3만 이하 24 16 3만 초과 4만 이하 27 18 4만 초과 30 20 비고 수소저장설비의 저장능력 Q=(10P+1)V1",
                "search_text": "FU671 저장능력 보호시설 안전거리 1만 이하 17 12 1만 초과 2만 이하 21 14 2만 초과 3만 이하 24 16 3만 초과 4만 이하 27 18 4만 초과 30 20",
            },
            {
                "hierarchy": "[FU671] 2 시설기준 > 2.9 피해저감설비기준 > 2.9.2 방호벽 설치",
                "page": 73,
                "content": "2.9.2 수소의 저장능력이 60m3 이상인 수소저장설비를 실내에 설치하는 경우 해당 공간의 벽은 방호벽으로 설치한다.",
                "search_text": "FU671 2.9.2 방호벽 수소저장설비 60m3 실내",
            },
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(message="수소 저장탱크와 건축물의 이격거리 기준을 알려줘")
    )

    assert response.intent == "fact"
    assert "10,000 이하: 17m | 12m" in response.answer
    assert "10,000 초과~20,000 이하: 21m | 14m" in response.answer
    assert "Q=(10P+1)V₁" in response.answer
    assert "모든 건축물에 일괄 적용" in response.answer
    assert "건축물 개구부" in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FU671"] * 7
    assert response.citations[2].hierarchy.endswith("표 2.1.2 보호시설과의 안전거리")


def test_fp217_storage_station_clearance_uses_station_rule_when_facility_is_named():
    chunks = [
        {
            "doc_type": "CODE",
            "doc_code": "FP217",
            "chunk_id": 1,
            "page": 17,
            "hierarchy": "[FP217] 2 시설기준 > 2.1 배치기준 > 2.1.1 보호시설과의 거리",
            "content": (
                "2.1.1.1 처리설비 및 저장설비는 외면으로부터 보호시설까지 표에서 정한 거리 이상을 유지한다. "
                "처리능력 및 저장능력 제1종보호시설 제2종보호시설 1만 이하 17m 12m "
                "1만 초과 2만 이하 21m 14m 2만 초과 3만 이하 24m 16m "
                "3만 초과 4만 이하 27m 18m 4만 초과 5만 이하 30m 20m "
                "5만 초과 99만 이하 30m (저온저장탱크 식) 20m 99만 초과 30m (저온저장탱크) 20m "
                "사업소 안 및 전용공업지역 안의 보호시설은 제외한다. 2.1.1.2 30m 이내 방호벽"
            ),
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP217",
            "chunk_id": 2,
            "page": 14,
            "hierarchy": "[FP217] 1 일반사항 > 1.3 용어정의 > 1.3.15.1 제1종보호시설",
            "content": "학교·병원·사람을 수용하는 연면적 1000m² 이상 건축물",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP217",
            "chunk_id": 3,
            "page": 14,
            "hierarchy": "[FP217] 1 일반사항 > 1.3 용어정의 > 1.3.15.2 제2종보호시설",
            "content": "주택 및 연면적 100m² 이상 1000m² 미만 건축물",
        },
    ]

    result = RagPipeline._fp217_storage_building_clearance(
        "저장식 수소연료 충전소 저장탱크와 건축물의 이격거리", [], chunks
    )

    assert result is not None
    sources, answer = result
    assert "FP217 2.1.1.1" in answer
    assert "10,000m³ 이하: 제1종 17m, 제2종 12m" in answer
    assert "30m 이내" in answer
    assert [item[0]["chunk_id"] for item in sources] == [1, 2, 3]


@pytest.mark.asyncio
async def test_fs551_user_side_scope_question_compares_supply_and_use_facility_clauses():
    query = "KGS FS551은 도시가스사업자 배관에만 적용돼, 가스사용자 건물 안 배관도 포함돼?"
    scope_chunks = [
        {
            "document_id": 1,
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "일반도시가스사업 제조소 및 공급 배관 기준",
            "chunk_id": 10,
            "hierarchy": "[FS551] 1 일반사항 > 1.1 적용범위",
            "page": 12,
            "filename": "FS551.pdf",
            "score": 1.0,
            "content": (
                "이 기준은 도시가스사업법 제2조제4호 및 제5호에 따른 일반도시가스사업의 "
                "가스공급시설 중 가스배관의 시설·기술·검사 및 진단에 적용한다."
            ),
        },
        {
            "document_id": 2,
            "doc_type": "CODE",
            "doc_code": "FU551",
            "title": "도시가스 사용시설 기준",
            "chunk_id": 20,
            "hierarchy": "[FU551] 1 일반사항 > 1.1 적용 범위",
            "page": 13,
            "filename": "FU551.pdf",
            "score": 1.0,
            "content": (
                "이 기준은 도시가스사업법 제2조제6호에 따른 가스사용시설의 "
                "설치·운영 및 검사에 적용한다."
            ),
        },
    ]

    class ScopeDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_document_scopes(self, codes):
            assert codes == ["FS551", "FU551"]
            return scope_chunks

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="unused",
            service_hub_fast_model="unused",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        ScopeDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "FS551 1.1" in response.answer and "FU551 1.1" in response.answer
    assert "사용자 측 내부배관 전체까지 자동으로 넓혀 읽을 수는 없습니다" in response.answer
    assert "물리적 인계점을 지정하지는 않습니다" in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FS551", "FU551"]
    assert [citation.page for citation in response.citations] == [12, 13]


def test_fs551_scope_and_user_supply_definition_are_kept_as_separate_evidence():
    scope = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 31,
        "hierarchy": "[FS551] 1 일반사항 > 1.1 적용범위",
        "page": 12,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": "이 기준은 일반도시가스사업의 가스공급시설 중 가스배관의 시설·기술·검사 및 진단에 적용한다.",
    }
    definition = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 32,
        "hierarchy": "[FS551] 1.3.4 사용자공급관",
        "page": 13,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": "1.3.4 사용자공급관이란 1.3.3(1)에 따른 공급관 중 가스사용자의 토지 경계에서 계량기 전단밸브까지 이르는 배관을 말한다.",
    }

    class ScopeDefinitionDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_document_scopes(self, codes):
            assert codes == ["FS551"]
            return [scope]

        def search_headings(self, heading, codes, limit=8):
            assert heading == "사용자공급관"
            assert codes == ["FS551"]
            assert limit == 100
            return [definition]

        def save_exchange(self, *args):
            return 2

    response = asyncio.run(
        RagPipeline(
            SimpleNamespace(service_hub_model="unused"),
            ScopeDefinitionDatabase(),
            ReasonerMustNotRun(),
        ).run(
            ChatRequest(
                message="KGS FS551 1.3.4 사용자공급관의 정의와 FS551 적용범위를 연결해서 설명해줘."
            )
        )
    )

    assert response.intent == "comparison"
    assert "FS551 1.1" in response.answer
    assert "FS551 1.3.4" in response.answer
    assert "모든 사용자 측 내부배관" in response.answer
    assert [citation.chunk_id for citation in response.citations] == [31, 32]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "codes"),
    [
        (
            "가스 배관 기밀시험의 목적, 시험 매체, 합격 기준을 간단히 알려줘.",
            ["FU671", "FP216", "FP217", "FS551", "FU551"],
        ),
        (
            "수소충전소 기밀시험에서 용적 12m³일 때 유지시간과 시험가스가 궁금해.",
            ["FU671", "FP216", "FP217"],
        ),
    ],
)
async def test_unscoped_gas_tightness_question_requests_facility_before_answering(
    query: str, codes: list[str]
):
    scope_rows = [
        {
            "document_id": index,
            "doc_type": "CODE",
            "doc_code": code,
            "title": f"{code} 기준",
            "chunk_id": 100 + index,
            "hierarchy": f"[{code}] 1 일반사항 > 1.1 적용범위",
            "page": index + 10,
            "filename": f"{code}.pdf",
            "score": 1.0,
            "content": f"{code}의 적용 시설 범위입니다.",
        }
        for index, code in enumerate(codes, start=1)
    ]

    class AmbiguousGasScopeDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set()

        def search_document_scopes(self, requested_codes):
            assert requested_codes == codes
            return scope_rows

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        AmbiguousGasScopeDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "clarification"
    assert "한 조항을 특정하지 않겠습니다" in response.answer
    assert "제조식 수소연료 충전시설" in response.answer
    if "FU551" in codes:
        assert "도시가스 사용시설" in response.answer
    assert [citation.doc_code for citation in response.citations] == codes


@pytest.mark.asyncio
async def test_unscoped_hydrogen_pipeline_question_clarifies_scope_before_retrieval():
    query = "수소배관망 압력저하가 생기면 우선 확인할 항목을 알려줘."
    codes = ["FP111", "FP216", "FP217", "FU671"]
    descriptions = {
        "FP111": "고압가스 특정제조시설",
        "FP216": "제조식 수소연료 충전시설",
        "FP217": "저장식 수소연료 충전시설",
        "FU671": "수소연료사용시설",
    }
    scope_rows = [
        {
            "document_id": index,
            "doc_type": "CODE",
            "doc_code": code,
            "title": f"{code} 기준",
            "chunk_id": 300 + index,
            "hierarchy": f"[{code}] 1 일반사항 > 1.1 적용범위",
            "page": index + 10,
            "filename": f"{code}.pdf",
            "score": 1.0,
            "content": f"{code} 적용범위: {descriptions[code]}에 적용한다.",
        }
        for index, code in enumerate(codes, start=1)
    ]

    class HydrogenPipelineScopeDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, requested_codes):
            return set()

        def search_document_scopes(self, requested_codes):
            assert requested_codes == codes
            return scope_rows

        def save_exchange(self, *args):
            return 20

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        HydrogenPipelineScopeDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "clarification"
    assert "적용 KGS 기준을 특정하지 않겠습니다" in response.answer
    assert "별도 수송·공급 배관망" in response.answer
    assert all(code in response.answer for code in codes)
    assert [citation.doc_code for citation in response.citations] == codes
    assert [citation.number for citation in response.citations] == [1, 2, 3, 4]
    assert "FS334" not in response.answer and "GC253" not in response.answer


@pytest.mark.asyncio
async def test_fp216_fp217_scope_comparison_uses_exact_application_clauses():
    query = (
        "KGS FP216과 FP217의 적용범위 조항을 비교해 수소 공급경로의 공통점과 차이를 설명해줘. "
        "그 문언만으로 확정할 수 없는 설계 책임이나 배관 경계는 추론으로 단정하지 말아줘."
    )
    scope_rows = [
        {
            "document_id": 1,
            "doc_type": "CODE",
            "doc_code": "FP216",
            "title": "제조식 수소연료 충전 기준",
            "chunk_id": 401,
            "hierarchy": "[FP216] 1 일반사항 > 1.1 적용범위",
            "page": 13,
            "filename": "FP216.pdf",
            "score": 1.0,
            "content": (
                "이 기준은 고압가스 제조시설 중 수소를 제조·압축하여 이동수단에 충전하는 "
                "제조식 수소연료 충전시설의 시설·기술·검사에 적용한다."
            ),
        },
        {
            "document_id": 2,
            "doc_type": "CODE",
            "doc_code": "FP217",
            "title": "저장식 수소연료 충전 기준",
            "chunk_id": 402,
            "hierarchy": "[FP217] 1 일반사항 > 1.1 적용범위",
            "page": 12,
            "filename": "FP217.pdf",
            "score": 1.0,
            "content": (
                "이 기준은 고압가스 충전시설 중 배관 또는 저장설비로부터 공급받은 수소를 압축하여 "
                "이동수단에 충전하는 저장식 수소연료 충전시설의 시설·기술·검사에 적용한다."
            ),
        },
    ]

    class HydrogenScopeComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code in {"FP216", "FP217"}}

        def search_headings(self, _heading, _codes, limit=8):
            assert limit == 100
            return []

        def search_document_scopes(self, codes):
            assert codes == ["FP216", "FP217"]
            return scope_rows

        def save_exchange(self, *args):
            return 22

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        HydrogenScopeComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "comparison"
    assert "공통점" in response.answer
    assert "FP216(제조식)" in response.answer and "수소를 제조·압축" in response.answer
    assert "FP217(저장식)" in response.answer and "배관 또는 저장설비에서 공급받은" in response.answer
    assert "배관의 정확한 물리적 경계까지 확정할 수 없습니다" in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FP216", "FP217"]
    assert [citation.page for citation in response.citations] == [13, 12]


def test_fp111_fp216_scope_comparison_does_not_auto_include_every_hydrogen_pipe():
    query = "KGS FP111과 FP216의 적용범위를 비교하고 수소 배관이 두 기준에 자동으로 포함되는지 알려줘."
    rows = [
        {
            "doc_type": "CODE",
            "doc_code": "FP111",
            "chunk_id": 501,
            "hierarchy": "[FP111] 1 일반사항 > 1.1 적용범위",
            "page": 13,
            "content": "이 기준은 고압가스 특정제조의 시설·기술·검사·감리에 적용한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP216",
            "chunk_id": 502,
            "hierarchy": "[FP216] 1 일반사항 > 1.1 적용범위",
            "page": 13,
            "content": "이 기준은 수소를 제조·압축하여 이동수단에 충전하는 제조식 수소연료 충전시설에 적용한다.",
        },
    ]

    result = RagPipeline._fp111_fp216_scope_comparison(query, ["FP111", "FP216"], rows)

    assert result is not None
    sources, answer = result
    assert [source["doc_code"] for source in sources] == ["FP111", "FP216"]
    assert "고압가스 특정제조" in answer
    assert "제조식 수소연료 충전시설" in answer
    assert "모든 수소 배관이 두 기준에 자동 포함" in answer
    assert "물리적 인계점" in answer


def test_combined_fp111_scope_and_procedure_outline_keeps_both_evidence_sets():
    query = "KGS FP111의 적용범위와 주요 검사방법을 정리해줘."
    scope = {
        "doc_type": "CODE",
        "doc_code": "FP111",
        "chunk_id": 601,
        "hierarchy": "[FP111] 1 일반사항 > 1.1 적용범위",
        "page": 13,
        "content": "이 기준은 고압가스 특정제조의 시설·기술·검사에 적용한다.",
    }
    procedures = [
        {
            "doc_type": "CODE",
            "doc_code": "FP111",
            "chunk_id": 602,
            "hierarchy": "[FP111] 4 검사기준 > 4.1 검사대상 > 4.1.1 중간검사",
            "page": 138,
            "content": "4.1.1 중간검사는 주요 설비의 검사대상을 확인한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP111",
            "chunk_id": 603,
            "hierarchy": "[FP111] 4 검사기준 > 4.2 검사방법 > 4.2.2 완성검사",
            "page": 147,
            "content": "4.2.2 완성검사는 시설의 시공 상태와 검사방법을 확인한다.",
        },
    ]

    result = RagPipeline._document_scope_and_inspection_outline(
        query, "FP111", [scope], procedures
    )

    assert result is not None
    _scope, answer, sources = result
    assert "적용범위(1.1)" in answer
    assert "4.1.1" in answer and "4.2.2" in answer
    assert [item["chunk_id"] for item in sources] == [602, 603]


def test_fp111_periodic_inspection_interval_preserves_special_condition_and_unknown_branch():
    query = "KGS FP111 정기검사 주기가 몇 년인지 조건별로 구분해줘."
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FP111",
        "chunk_id": 604,
        "hierarchy": "[FP111] 4 검사기준 > 4.1 검사항목 > 4.1.3 정기검사",
        "page": 139,
        "content": (
            "정기검사 항목은 다음과 같다. 다만, 규칙 제3조제3호에 해당하는 시설은 "
            "최초완성검사를 받은 날로부터 10년이 되는 날 및 그 이후 매 4년이 경과한 날에 속하는 연도에 "
            "정기검사를 하는 경우에는 (1)에서 (79)까지 검사하고, 그 외의 연도에 정기검사를 하는 경우에는 "
            "(1)에서 (78)까지 검사를 한다."
        ),
    }

    result = RagPipeline._fp111_periodic_inspection_interval(query, [chunk])

    assert result is not None
    _source, answer, excerpt = result
    assert "10년" in answer and "매 4년" in answer
    assert "그 외 시설의 별도 검사주기를 확정하지" in answer
    assert "10년" in excerpt and "4년" in excerpt


@pytest.mark.asyncio
async def test_fp216_fp217_tightness_media_and_acceptance_comparison_uses_both_sources():
    chunks = {
        code: {
            "document_id": index,
            "doc_type": "CODE",
            "doc_code": code,
            "title": f"{code} 수소충전시설 기준",
            "chunk_id": 2100 + index,
            "hierarchy": f"[{code}] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
            "page": 95 if code == "FP217" else 111,
            "filename": f"{code}.pdf",
            "score": 1.0,
            "content": (
                "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
                "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 해당 고압가스설비에 "
                "저장 또는 처리되는 가스를 사용하여 기밀시험을 할 수 있다. 이 경우 압력은 "
                "단계적으로 올려 이상이 없는지 확인하면서 승압한다. "
                "(5) 기밀시험은 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다."
            ),
        }
        for index, code in enumerate(("FP216", "FP217"), start=1)
    }

    class HydrogenTightnessComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes_to_check):
            return {code for code in codes_to_check if code in chunks}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert limit == 100
            return [chunks[code] for code in codes if code in chunks]

        def search_document_scopes(self, _codes):
            return []

        def save_exchange(self, *args):
            return 4

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        HydrogenTightnessComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="KGS FP216과 FP217의 기밀시험 매체와 합격 기준 차이를 비교해줘."
        )
    )

    assert response.intent == "comparison"
    assert "이 두 항목에 한해 차이가 없습니다" in response.answer
    assert "FP216" in response.answer and "FP217" in response.answer
    assert [citation.doc_code for citation in response.citations] == [
        "FP216", "FP216", "FP217", "FP217"
    ]
    assert [citation.page for citation in response.citations] == [111, 112, 95, 96]
    assert [citation.number for citation in response.citations] == [1, 2, 3, 4]


def test_fs551_fp217_tightness_comparison_keeps_distinct_hold_time_rules():
    chunks = [
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 2201,
            "content": "기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 2202,
            "content": "매설된 배관은 시험가스를 넣어서 12시간 경과한 후 판정한다. 매설된 배관은 시험가스를 넣어 24시간 경과한 후 판정한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP217",
            "chunk_id": 2203,
            "content": "기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FP217",
            "chunk_id": 2204,
            "content": "1m3 미만 48분 1m3 이상 10m3 미만 480분 10m3 이상 48×V분",
        },
    ]
    result = RagPipeline._fs551_fp217_tightness_comparison(
        "KGS FS551과 FP217의 기밀시험 시험가스와 유지시간 기준을 비교해줘.",
        chunks,
    )

    assert result is not None
    sources, answer = result
    assert [row[0]["doc_code"] for row in sources] == ["FS551", "FS551", "FP217", "FP217"]
    assert "FS551의 12·24시간 조건" in answer
    assert "FP217 유지시간 표" in answer


def test_fs551_new_pipe_tightness_methods_are_split_into_four_methods():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 2205,
        "hierarchy": "[FS551] 4.2.2.9.4 신규 배관",
        "content": (
            "4.2.2.9.4 신규 배관 (1) 발포액을 이음부에 도포하여 거품의 발생 여부로 판정하는 방법 "
            "(2) 시험가스 농도가 0.2% 이하에서 작동하는 가스검지기를 사용한다. 매설된 배관은 12시간 후 판정한다. "
            "(3) 고압·중압 용접 배관은 방사선투과시험 합격 후 통과가스를 사용하고 매설배관은 24시간 후 판정한다. "
            "(4) 압력측정기구와 용적·최고사용압력에 따른 기밀유지시간 이상을 유지한다."
        ),
    }
    result = RagPipeline._fs551_new_pipe_tightness_methods(
        "KGS FS551 신규 설치 배관 기밀시험 방법을 단계별로 발포액, 가스검지기, 압력측정기구로 구분해줘.",
        [chunk],
    )

    assert result is not None
    _source, answer, excerpt = result
    assert "네 가지" in answer
    assert "12시간" in answer and "24시간" in answer
    assert "압력측정기구" in answer and "0.2%" in excerpt


def test_fs551_tightness_method_summary_includes_acceptance_criterion():
    chunks = [
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 2206,
            "hierarchy": "[FS551] 4.2.2.9.3",
            "page": 94,
            "content": "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. (2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다.",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 2207,
            "hierarchy": "[FS551] 4.2.2.9.4",
            "page": 95,
            "content": "4.2.2.9.4 신규 배관 (1) 발포액 방법 (2) 가스검지기 방법 (3) 용접·방사선투과시험 합격 배관 방법 (4) 압력측정기구 방법",
        },
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 2208,
            "hierarchy": "[FS551] 95페이지",
            "page": 95,
            "content": "(4) 기밀시험은 기밀시험압력에서 누출 등의 이상이 없을 때 합격으로 한다.",
        },
    ]
    result = RagPipeline._fs551_tightness_method_acceptance_summary(
        "KGS FS551 기밀시험 방법과 합격 판정기준을 알려줘.", chunks
    )

    assert result is not None
    sources, answer = result
    assert [row[0]["chunk_id"] for row in sources] == [2206, 2207, 2208]
    assert "합격 판정" in answer and "누출 등의 이상이 없을 때" in answer


@pytest.mark.asyncio
async def test_fu671_tightness_media_and_acceptance_use_only_the_tightness_clause():
    query = "KGS FU671 수소연료사용시설의 배관 기밀시험 매체와 합격 기준은?"
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 991,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "수소가스설비와 배관의 기밀시험은 다음 기준에 따라 실시한다. "
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(2) 그 설비가 취성 파괴를 일으킬 우려가 없는 온도에서 한다. "
            "(3) 기밀시험압력은 상용압력 이상으로 한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 수소를 사용하여 "
            "기밀시험을 할 수 있다. 이 경우 압력은 단계적으로 올린다. "
            "(5) 기밀시험은 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다. "
            "(6) 인원은 최소인원으로 한다."
        ),
    }

    class Fu671TightnessDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FU671"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        Fu671TightnessDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "공기 또는 위험성이 없는 기체" in response.answer
    assert "수소를 사용하여 기밀시험" in response.answer
    assert "단계적으로 올린다" in response.answer
    assert "기밀시험압력에서 누설 등의 이상이 없을 때 합격" in response.answer
    assert "취성 파괴" not in response.answer
    assert "최소인원" not in response.answer
    assert len(response.citations) == 1 and response.citations[0].page == 95
    assert "수소를 사용하여 기밀시험" in response.citations[0].excerpt
    assert "누설 등의 이상이 없을 때 합격" in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fu671_pressure_drop_is_not_mistaken_for_a_leak_and_separates_general_checks():
    query = (
        "KGS FU671에서 수소가스설비 또는 배관의 압력저하가 발생했다는 이유만으로 "
        "누출이라고 단정할 수 있어? FU671에서 직접 요구하는 기밀시험 기준과, "
        "사고 원인 진단을 위해 추가로 확인해야 하는 정보를 구분해줘."
    )
    chunk = {
        "document_id": 4,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1002,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(3) 기밀시험압력은 상용압력 이상으로 한다. 이 경우 시험할 부분의 용적에 대응한 "
            "기밀유지시간 이상을 유지하고 처음과 마지막 시험의 측정압력차가 압력측정기구의 "
            "허용오차 안에 있는 것을 확인한다. 처음과 마지막 시험의 온도차가 있는 경우에는 "
            "압력차를 보정한다. "
            "(5) 기밀시험은 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다."
        ),
    }

    class Fu671PressureDropDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FU671"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 21

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        Fu671PressureDropDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "fact"
    assert "압력저하만으로 누출이라고 단정할 수는 없습니다" in response.answer
    assert "정식 기밀시험의 조건·판정기준" in response.answer
    assert "기밀유지시간 이상을 유지하고 처음과 마지막 시험의 측정압력차" in response.answer
    assert "처음과 마지막 시험의 측정압력차" in response.answer
    assert "온도차가 있는 경우에는 압력차를 보정" in response.answer
    assert "일반적인 원인분석에서 추가로 확인할 자료" in response.answer
    assert "이 기준의 별도 의무사항이라는 뜻은 아님" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_code == "FU671"
    assert response.citations[0].page == 95
    assert "측정압력차" in response.citations[0].excerpt
    assert "누설 등의 이상이 없을 때 합격" in response.citations[0].excerpt
    assert "4.2.1.5.3" not in response.citations[0].excerpt


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("data/saga.db").exists(), reason="local private corpus index is not part of the public repository")
async def test_cross_code_pressure_drop_keeps_fu671_and_fs551_rules_separate():
    query = (
        "KGS FS551과 FU671에서 운전 중 압력저하가 발생했을 때 바로 배관 누출로 "
        "단정해도 되는지, 확인 순서를 근거와 함께 설명해줘."
    )

    class CrossCodeDatabase(Database):
        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="unused",
            service_hub_fast_model="unused",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        CrossCodeDatabase(Path("data/saga.db")),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "fact"
    assert "압력저하만으로 누출이라고 단정할 수는 없습니다" in response.answer
    assert "FS551도 내압시험에서 압력강하·이상변형·파손이 없는지 확인" in response.answer
    assert {citation.doc_code for citation in response.citations} == {"FS551", "FU671"}
    assert {citation.page for citation in response.citations} == {95, 98}


@pytest.mark.asyncio
async def test_hydrogen_facility_choice_resolves_clarification_and_answers_concisely():
    chunk = {
        "document_id": 2,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소자동차 충전의 시설·기술·검사 기준",
        "chunk_id": 992,
        "hierarchy": "[FP217] 4 검사기준 > 4.2 검사방법 > 4.2.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "배관의 기밀시험은 다음 기준에 따라 실시한다. "
            "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체의 압력으로 실시한다. "
            "(2) 설비가 취성 파괴를 일으킬 우려가 없는 온도에서 한다. "
            "(3) 시험압력은 상용압력 이상으로 한다. "
            "(4) 검사 상황에 따라 위험이 없다고 판단되는 경우에는 해당 고압가스설비에 "
            "저장 또는 처리되는 가스를 사용하여 기밀시험을 할 수 있으며 압력은 단계적으로 올린다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다. "
            "(6) 시험 중에는 출입을 제한하고 주변을 통제한다."
        ),
    }
    clarification = (
        "수소 시설이라면 사용시설인지, 제조식 충전소인지, 저장식 충전소인지 "
        "또는 적용 KGS 번호를 지정해 주세요."
    )
    initial_question = "수소충전소 배관 기밀시험의 매체와 합격 기준은?"
    history = [
        {"role": "user", "content": initial_question},
        {"role": "assistant", "content": clarification},
    ]
    assert RagPipeline._selected_hydrogen_facility_tightness_code("저장식이야.", history) == "FP217"
    time_and_gas_question_history = [
        {
            "role": "user",
            "content": "수소충전소 기밀시험에서 용적 12m³일 때 유지시간과 시험가스가 궁금해.",
        },
        {"role": "assistant", "content": clarification},
    ]
    assert (
        RagPipeline._selected_hydrogen_facility_tightness_code(
            "저장식이야.", time_and_gas_question_history
        )
        == "FP217"
    )

    class ContextualHydrogenDatabase:
        def history(self, _conversation_id):
            return history

        def existing_document_codes(self, _codes):
            return set()

        def search_document_scopes(self, _codes):
            return []

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 2

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualHydrogenDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message="저장식이야.", conversation_id="h2-scope-choice"))

    assert response.intent == "fact"
    assert "공기 또는 위험성이 없는 불활성기체" in response.answer
    assert "저장 또는 처리되는 가스를 사용하여 기밀시험" in response.answer
    assert "단계적으로 올린다" in response.answer
    assert "누설 등의 이상이 없을 때 합격" in response.answer
    assert "출입을 제한" not in response.answer
    assert len(response.citations) == 2
    assert [citation.doc_code for citation in response.citations] == ["FP217", "FP217"]
    assert [citation.page for citation in response.citations] == [95, 96]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "history"),
    [
        (
            "KGS FP217 기밀시험 용적 12m³일 때 유지시간과 시험가스도 궁금해.",
            [],
        ),
        (
            "KGS FP217 기밀시험 용적 12m³일 때 최소 유지시간과 시험가스를 알려줘.",
            [],
        ),
        (
            "저장식이야.",
            [
                {
                    "role": "user",
                    "content": "수소충전소 기밀시험에서 용적 12m³일 때 유지시간과 시험가스가 궁금해.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "수소 시설이라면 사용시설인지, 제조식 충전소인지, 저장식 충전소인지 "
                        "또는 적용 KGS 번호를 지정해 주세요."
                    ),
                },
            ],
        ),
    ],
)
async def test_hydrogen_hold_time_and_test_gas_are_both_answered(
    message: str, history: list[dict[str, str]]
):
    chunk = {
        "document_id": 3,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 993,
        "hierarchy": "[FP217] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체로 실시한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 해당 고압가스설비에 "
            "저장 또는 처리되는 가스를 사용하여 기밀시험을 할 수 있다. 이 경우 압력은 "
            "단계적으로 올려 이상이 없는지 확인하면서 승압한다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다. "
            "표 4.2.1.5.2 시험 용적에 따른 기밀유지시간"
        ),
    }

    class CombinedHydrogenQuestionDatabase:
        def history(self, _conversation_id):
            return history

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FP217"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def search_document_scopes(self, _codes):
            return []

        def save_exchange(self, *args):
            return 3

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        CombinedHydrogenQuestionDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=message, conversation_id="combined-h2-tightness"))

    assert response.intent == "fact"
    assert "576분" in response.answer
    assert "저장 또는 처리되는 가스" in response.answer
    assert "단계적으로 올" in response.answer
    assert [citation.page for citation in response.citations] == [96, 95, 96]
    assert [citation.number for citation in response.citations] == [1, 2, 3]


@pytest.mark.asyncio
async def test_hydrogen_hold_time_followup_prefers_current_duration_over_prior_answer():
    chunk = {
        "document_id": 3,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 994,
        "hierarchy": "[FP217] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "page": 96,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "표 4.2.1.5.2 시험 용적에 따른 기밀유지시간. "
            "10㎥ 이상 48×V분. 다만, 2880분을 초과한 경우는 2880분으로 할 수 있다. "
            "[비고] V는 피시험부분의 용적(단위 : ㎥)이다."
        ),
    }

    class ContextualFP217Database:
        def history(self, _conversation_id):
            return [
                {"role": "user", "content": "KGS FP217 기밀시험 용적 12m³의 유지시간은?"},
                {"role": "assistant", "content": "FP217의 계산 결과는 576분입니다."},
            ]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FP217"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def search_document_scopes(self, _codes):
            return []

        def save_exchange(self, *args):
            return 4

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualFP217Database(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            conversation_id="fp217-current-duration",
            message="그럼 25시간 유지하면 합격이야?",
        )
    )

    assert "실제 유지시간 25시간 (1500분)" in response.answer
    assert "실제 유지시간 576분" not in response.answer
    assert "누설 등 다른 판정조건" in response.answer


@pytest.mark.asyncio
async def test_hydrogen_hold_time_volume_only_followup_reuses_prior_standard():
    chunk = {
        "document_id": 31,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1031,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간",
    }

    class ContextualFU671VolumeDatabase:
        def history(self, _conversation_id):
            return [
                {
                    "role": "user",
                    "content": "KGS FU671 기밀시험 용적 1m³일 때 기밀유지시간은?",
                },
                {"role": "assistant", "content": "V=1m³이면 240분입니다."},
            ]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FU671"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 31

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualFU671VolumeDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 10m³이면?",
            conversation_id="fu671-volume-only-followup",
        )
    )

    assert response.intent == "fact"
    assert "24×10=240분" in response.answer
    assert "V=1㎥" not in response.answer
    assert response.citations[0].doc_code == "FU671"
    assert response.citations[0].page == 95
    assert "4.2.2.9.3" in response.citations[0].hierarchy


@pytest.mark.asyncio
async def test_fu671_clarification_followup_answers_hold_time_and_hydrogen_test_gas():
    history = [
        {
            "role": "user",
            "content": "수소충전소 기밀시험에서 용적 12m³일 때 유지시간과 시험가스가 궁금해.",
        },
        {
            "role": "assistant",
            "content": (
                "수소 시설이라면 사용시설인지, 제조식 충전소인지, 저장식 충전소인지 "
                "또는 적용 KGS 번호를 지정해 주세요."
            ),
        },
    ]
    chunk = {
        "document_id": 4,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 994,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 수소를 사용하여 "
            "기밀시험을 할 수 있다. 이 경우 압력은 단계적으로 올려 이상이 없음을 확인하면서 승압한다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다. "
            "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간"
        ),
    }

    class FU671ContextualHoldTimeDatabase:
        def history(self, _conversation_id):
            return history

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FU671"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def search_document_scopes(self, _codes):
            return []

        def save_exchange(self, *args):
            return 4

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FU671ContextualHoldTimeDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message="수소연료사용시설이야.", conversation_id="fu671-contextual-hold-time")
    )

    assert response.intent == "fact"
    assert "24×12=288분" in response.answer
    assert "기본 시험 매체" in response.answer
    assert "원칙적으로 공기 또는 위험성이 없는 기체" in response.answer
    assert "수소를 기밀시험 가스로 사용할 수 있습니다" in response.answer
    assert "단계적으로 올" in response.answer
    assert [citation.page for citation in response.citations] == [95, 95]
    assert [citation.number for citation in response.citations] == [1, 2]
    assert "4.2.2.9.3(1) 기본 시험 매체" in response.citations[1].hierarchy
    assert "4.2.2.9.3(4) 수소 시험가스 사용 조건" in response.citations[1].hierarchy
    assert "(1)" in response.citations[1].excerpt and "(4)" in response.citations[1].excerpt


@pytest.mark.asyncio
async def test_natural_storage_station_phrase_selects_fp217_without_llm_router():
    chunk = {
        "document_id": 5,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 995,
        "hierarchy": "[FP217] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체로 실시한다. "
            "(2) 적절한 온도에서 한다. (3) 시험압력 조건이다. "
            "(4) 위험이 없다고 판단되는 경우 해당 고압가스설비에 저장 또는 처리되는 가스를 "
            "사용할 수 있다. 압력은 단계적으로 올린다. "
            "(5) 누설 등의 이상이 없을 때 합격으로 한다."
        ),
    }

    class NaturalFacilityDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, _codes):
            return set()

        def search_document_scopes(self, _codes):
            return []

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 5

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        NaturalFacilityDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message="저장식 수소충전소 배관 기밀시험의 매체와 합격 기준은?")
    )

    assert response.intent == "fact"
    assert [citation.doc_code for citation in response.citations] == ["FP217", "FP217"]
    assert [citation.page for citation in response.citations] == [95, 96]
    assert "위험성이 없는 기체" in response.answer


def test_fp216_tightness_summary_splits_sources_at_the_actual_pdf_page_break():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FP216",
        "hierarchy": "[FP216] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(2) 취성파괴 우려가 없는 온도에서 한다. (3) 상용압력 조건. "
            "(4) 검사 상황에서 위험이 없다고 판단되면 해당 고압가스설비에 저장 또는 처리되는 "
            "가스로 시험할 수 있다. 압력은 단계적으로 올린다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다."
        ),
    }

    result = RagPipeline._hydrogen_tightness_summary_with_page_spans(
        "KGS FP216 기밀시험 매체와 합격 기준은?", [chunk]
    )

    assert result is not None
    _source, answer, page_spans = result
    assert [page for page, _hierarchy, _excerpt in page_spans] == [111, 112]
    assert "시험 매체" in answer and "합격 기준" in answer
    assert "저장 또는 처리되는" in page_spans[1][2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "volume", "expected_minutes", "expected_page", "expected_marker"),
    [
        ("FP217", "0.5m³", "48분", 96, "2880분"),
        ("FP217", "12m³", "576분", 96, "2880분"),
        ("FP216", "70m³", "2880분", 111, "3360분"),
        ("FU671", "12m³", "288분", 95, "1440분"),
    ],
)
async def test_hydrogen_tightness_hold_time_calculates_with_the_selected_code_table(
    code: str, volume: str, expected_minutes: str, expected_page: int, expected_marker: str
):
    section = "4.2.2.9.3" if code == "FU671" else "4.2.1.5.2"
    chunk = {
        "document_id": 10,
        "doc_type": "CODE",
        "doc_code": code,
        "title": f"{code} 기준",
        "chunk_id": 1000,
        "hierarchy": f"[{code}] 4 검사기준 > {section} 기밀시험방법",
        "page": expected_page - 1 if code == "FP217" else expected_page,
        "filename": f"{code}.pdf",
        "score": 1.0,
        # The PDF text index retains the table heading but drops its drawn cell values.
        "content": f"{section} 기밀시험방법. 표 {section} 시험 용적에 따른 기밀유지시간",
    }

    class HydrogenHoldTimeDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code} if code in codes else set()

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == [code]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 10

    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        HydrogenHoldTimeDatabase(),
        ReasonerMustNotRun(),
    )
    response = await pipeline.run(
        ChatRequest(
            message=f"KGS {code} 기밀시험 용적 {volume}일 때 최소 유지시간과 상한 조건을 계산해줘."
        )
    )

    assert response.intent == "fact"
    assert expected_minutes in response.answer
    assert expected_marker in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_code == code
    assert response.citations[0].page == expected_page
    assert "기밀유지시간 이상을 유지" in response.citations[0].excerpt
    expected_cap = "1440분" if code == "FU671" else "2880분"
    assert f"{expected_cap}을 초과한 경우는 {expected_cap}으로 할 수 있다" in response.citations[0].excerpt
    assert "48분(48분)" not in response.answer
    if code == "FP216":
        shorthand_response = await pipeline.run(
            ChatRequest(
                message=f"KGS {code} 피시험부분 용적이 {volume}이면 기밀유지시간은?"
            )
        )
        assert expected_minutes in shorthand_response.answer
        assert shorthand_response.citations[0].doc_code == code
    if code == "FU671":
        compliance_response = await pipeline.run(
            ChatRequest(
                message=(
                    "KGS FU671에서 기밀시험 용적 80m³에서 20시간만 유지하면 "
                    "합격으로 볼 수 있어? 표 계산값과 1440분 상한 예외도 구분해줘."
                )
            )
        )
        assert "24×80=1920분" in compliance_response.answer
        assert "상한 예외를 적용하면 최소 1440분" in compliance_response.answer
        assert "실제 유지시간 20시간 (1200분)" in compliance_response.answer
        assert "240분 부족" in compliance_response.answer
        assert "기밀유지시간 요건을 충족하지 않습니다" in compliance_response.answer

        cap_edge_response = await pipeline.run(
            ChatRequest(
                message=(
                    "KGS FU671 기밀시험 용적 80m³에서 25시간 유지했으면 시간 기준을 충족해? "
                    "24시간 상한 예외가 적용되는지도 알려줘."
                )
            )
        )
        assert "계산값 1920분" in cap_edge_response.answer
        assert "실제 유지시간 25시간 (1500분)" in cap_edge_response.answer
        assert "상한 예외로 1440분을 적용하는 경우에는 시간 요건을 충족합니다" in cap_edge_response.answer
        assert "상한 예외 적용 여부를 확인해야 합니다" in cap_edge_response.answer

        class ContextualHydrogenHoldTimeDatabase(HydrogenHoldTimeDatabase):
            def history(self, _conversation_id):
                return [
                    {
                        "role": "user",
                        "content": (
                            "KGS FU671 수소연료사용시설에서 기밀시험 용적이 80㎥일 때 "
                            "표상 유지시간과 상한 예외를 설명해줘."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "V=80㎥의 계산값은 1920분이고 1440분 상한을 적용할 수 있습니다.",
                    },
                ]

        contextual_cap_response = await RagPipeline(
            SimpleNamespace(service_hub_model="unused"),
            ContextualHydrogenHoldTimeDatabase(),
            ReasonerMustNotRun(),
        ).run(
            ChatRequest(
                message="그럼 25시간 유지하면 시간 기준은 충족하는 거야?",
                conversation_id="fu671-hold-time-contextual-followup",
            )
        )
        assert "계산값 1920분" in contextual_cap_response.answer
        assert "실제 유지시간 25시간 (1500분)" in contextual_cap_response.answer
        assert "상한 예외로 1440분을 적용하는 경우에는 시간 요건을 충족합니다" in contextual_cap_response.answer
        assert "상한 예외 적용 여부를 확인해야 합니다" in contextual_cap_response.answer
        assert contextual_cap_response.citations[0].doc_code == "FU671"


def test_hydrogen_tightness_hold_time_reports_all_requested_volumes():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FP217",
        "chunk_id": 1002,
        "hierarchy": "[FP217] 4.2.1.5.2 기밀시험방법",
        "page": 96,
        "content": "표 4.2.1.5.2 시험 용적에 따른 기밀유지시간",
    }
    result = RagPipeline._hydrogen_tightness_hold_time_calculation(
        "KGS FP217 기밀시험 용적 0.5m³, 5m³, 80m³의 최소 유지시간을 각각 계산해줘.",
        "FP217",
        [chunk],
    )

    assert result is not None
    _source, answer, _page, _hierarchy, _excerpt = result
    assert "V=0.5㎥" in answer and "최소 48분" in answer
    assert "V=5㎥" in answer and "최소 480분" in answer
    assert "V=80㎥" in answer and "3840분" in answer and "최소 2880분" in answer

    fu671_chunk = {
        "doc_type": "CODE",
        "doc_code": "FU671",
        "chunk_id": 1003,
        "hierarchy": "[FU671] 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "content": "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간",
    }
    fu671_result = RagPipeline._hydrogen_tightness_hold_time_calculation(
        "KGS FU671 기밀시험 용적 0.99m³, 1m³, 10m³, 60m³의 유지시간을 각각 계산해줘.",
        "FU671",
        [fu671_chunk],
    )
    assert fu671_result is not None
    _source, fu671_answer, _page, _hierarchy, fu671_excerpt = fu671_result
    assert "V=0.99㎥" in fu671_answer and "최소 24분" in fu671_answer
    assert "V=1㎥" in fu671_answer and "최소 240분" in fu671_answer
    assert "V=10㎥" in fu671_answer and "24×10=240분" in fu671_answer
    assert "V=60㎥" in fu671_answer and "24×60=1440분" in fu671_answer
    assert "1㎥ 미만 24분" in fu671_excerpt
    assert "48×V" not in fu671_answer and "48×V" not in fu671_excerpt


@pytest.mark.asyncio
@pytest.mark.parametrize("volume", ["0m³", "-1m³"])
async def test_hydrogen_tightness_hold_time_rejects_nonpositive_test_volume(volume: str):
    chunk = {
        "document_id": 4,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1001,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간. 1㎥ 미만 24분. "
            "[비고] V는 피시험부분의 용적(단위 : ㎥)이다."
        ),
    }

    class InvalidHydrogenVolumeDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FU671"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 11

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        InvalidHydrogenVolumeDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message=f"KGS FU671 기밀시험 용적 {volume}일 때 최소 유지시간은?")
    )

    assert response.intent == "fact"
    assert "기밀유지시간을 산정하지 않겠습니다" in response.answer
    assert "실제 시험구간의 용적을 확인" in response.answer
    assert "24분" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_code == "FU671"
    assert response.citations[0].page == 95
    assert "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간" in response.citations[0].excerpt
    assert "V는 피시험부분의 용적" in response.citations[0].excerpt
    assert "4.2.1.5.3" not in response.citations[0].excerpt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "KGS FU671에서 수소를 기밀시험 가스로 사용할 수 있는 조건은?",
        "수소연료사용시설에서 수소를 기밀시험 가스로 사용할 수 있는 조건은?",
        "KGS FU671 기밀시험의 기본 시험가스는 무엇이고 수소를 시험가스로 사용할 수 있는 조건은?",
        "KGS FU671 기밀시험의 기본 시험가스는 무엇이야?",
    ],
)
async def test_fu671_hydrogen_test_gas_question_uses_only_clause_four(query: str):
    chunk = {
        "document_id": 6,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 996,
        "hierarchy": "[FU671] 4 검사기준 > 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 수소를 사용하여 "
            "기밀시험을 할 수 있다. 이 경우 압력은 단계적으로 올려 이상이 없음을 확인하면서 승압한다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다. "
            "(6) 작업 인원을 최소로 한다."
        ),
    }

    class FU671GasTestDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {"FU671"} if "FU671" in codes else set()

        def search_document_scopes(self, _codes):
            return []

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FU671"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 6

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FU671GasTestDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "fact"
    assert "합격으로 한다" not in response.answer
    assert "작업 인원" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 95
    assert "(5)" not in response.citations[0].excerpt
    if "기본" in query:
        assert "기본 시험 매체" in response.answer
        assert "공기 또는 위험성이 없는 기체" in response.answer
        assert "원칙적으로 공기 또는 위험성이 없는 기체" in response.citations[0].excerpt
        assert "4.2.2.9.3(1) 기본 시험 매체" in response.citations[0].hierarchy
        if "수소를 시험가스로" in query:
            assert "위험이 없다고 판단되는 경우" in response.answer
            assert "단계적으로 올" in response.answer and "이상이 없는지 확인" in response.answer
            assert "(4)" in response.citations[0].excerpt
        else:
            assert "위험이 없다고 판단되는 경우" not in response.answer
            assert "(4)" not in response.citations[0].excerpt
    else:
        assert "위험이 없다고 판단되는 경우" in response.answer
        assert "단계적으로 올" in response.answer and "이상이 없는지 확인" in response.answer
        assert "기본 시험 매체" not in response.answer


@pytest.mark.asyncio
async def test_hydrogen_test_gas_cross_code_comparison_keeps_evidence_for_each_standard():
    fp217 = {
        "document_id": 7,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 997,
        "hierarchy": "[FP217] 4.2.1.5.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "(1) 공기 또는 위험성이 없는 기체로 시험한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 해당 고압가스설비에 "
            "저장 또는 처리되는 가스를 사용하여 기밀시험을 할 수 있다. 이 경우 압력은 "
            "단계적으로 올려 이상이 없음을 확인하면서 승압한다. (5) 합격 기준이다."
        ),
    }
    fu671 = {
        "document_id": 8,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 998,
        "hierarchy": "[FU671] 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "(1) 공기 또는 위험성이 없는 기체로 시험한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 수소를 사용하여 "
            "기밀시험을 할 수 있다. 이 경우 압력은 단계적으로 올려 이상이 없음을 확인하면서 승압한다. "
            "(5) 합격 기준이다."
        ),
    }

    class HydrogenComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code in {"FP217", "FU671"}}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217", "FU671"]
            assert limit == 100
            return [fp217, fu671]

        def save_exchange(self, *args):
            return 7

    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        HydrogenComparisonDatabase(),
        ReasonerMustNotRun(),
    )
    response = await pipeline.run(
        ChatRequest(
            message="KGS FP217과 FU671에서 수소로 기밀시험하는 조건은 같은가? 차이와 공통점을 비교해줘."
        )
    )

    assert response.intent == "comparison"
    assert "시험가스의 문언상 범위는 서로 다릅니다" in response.answer
    assert "FP217" in response.answer and "저장 또는 처리되는 가스" in response.answer
    assert "FU671" in response.answer and "시험가스는 수소" in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FP217", "FU671"]
    assert [citation.page for citation in response.citations] == [96, 95]

    fp216 = {
        **fp217,
        "document_id": 9,
        "doc_code": "FP216",
        "chunk_id": 999,
        "hierarchy": "[FP216] 4.2.1.5.2 기밀시험방법",
        "page": 111,
    }
    same_scope = pipeline._hydrogen_test_gas_comparison(
        "KGS FP216과 FP217에서 저장 또는 처리되는 가스로 기밀시험하는 조건은 동일해? 공통점과 차이를 비교해줘.",
        ["FP216", "FP217"],
        [fp216, fp217],
    )
    assert same_scope is not None
    same_scope_answer, same_scope_citations = same_scope
    assert "완전히 같은 조건은 아닙니다" not in same_scope_answer
    assert "같은 표현으로 규정되어 있습니다" in same_scope_answer
    assert [citation.page for citation in same_scope_citations] == [112, 96]
    assert pipeline._hydrogen_test_gas_comparison(
        (
            "KGS FP217과 FU671의 수소 기밀시험에서 시험가스 조건뿐 아니라 시험압력, "
            "유지시간, 합격판정도 비교해줘."
        ),
        ["FP217", "FU671"],
        [fp217, fu671],
    ) is None


def test_fp216_fp217_scope_comparison_does_not_hijack_test_detail_questions():
    codes = ["FP216", "FP217"]
    assert RagPipeline._asks_hydrogen_facility_scope_comparison(
        "KGS FP216과 FP217의 적용 대상과 공급 경로 차이를 비교해줘.", codes
    )
    assert not RagPipeline._asks_hydrogen_facility_scope_comparison(
        (
            "KGS FP216과 FP217의 배관 기밀시험을 비교해줘. 시험매체, 시험압력, "
            "유지시간, 합격판정을 나누고 적용 대상을 함께 알려줘."
        ),
        codes,
    )


@pytest.mark.asyncio
async def test_fp216_fp217_detailed_tightness_comparison_covers_all_requested_fields(
    tmp_path: Path,
):
    database = Database(tmp_path / "test.db")
    database.initialize()
    for code, scope_page, test_page, table_page, scope_content in (
        (
            "FP216",
            13,
            111,
            111,
            "고압가스 제조시설 중 수소를 제조·압축하여 이동수단에 충전하는 제조식 시설에 적용한다.",
        ),
        (
            "FP217",
            12,
            95,
            96,
            "배관 또는 저장설비에서 공급받은 수소를 압축하여 이동수단에 충전하는 저장식 시설에 적용한다.",
        ),
    ):
        clause_content = (
            "고압가스설비와 배관의 기밀시험은 다음 기준에 따라 실시한다. "
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(2) 설비가 취성 파괴를 일으킬 우려가 없는 온도에서 한다. "
            "(3) 기밀시험압력은 상용압력 이상으로 하되, 0.7MPa를 초과하는 경우 "
            "0.7MPa압력 이상으로 한다. 이 경우 표 4.2.1.5.2와 같이 시험할 부분의 용적에 대응한 "
            "기밀유지시간 이상을 유지하고 처음과 마지막 시험의 측정압력차가 압력측정기구의 "
            "허용오차 안에 있는 것을 확인한다. 처음과 마지막 시험의 온도차가 있는 경우에는 "
            "압력차를 보정한다. "
            + (
                "표 4.2.1.5.2 시험 용적에 따른 기밀유지시간 압력측정기구 용적 기밀유지시간 "
                "압력계 또는 자기압력기록계 1m3 미만 48분 1m3 이상 10m3 미만 480분 "
                "10m3 이상 48×V분 (다만, 2880분을 초과한 경우는 2880분으로 할 수 있다) "
                "[비고] V는 피시험부분의 용적(단위 : m3)이다. "
                if code == "FP216"
                else ""
            )
            + "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 저장 또는 처리되는 가스를 "
            "사용하여 기밀시험을 할 수 있다. 압력은 단계적으로 올린다. "
            "(5) 기밀시험은 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다."
        )
        rows = [
            {
                "hierarchy": f"[{code}] 1 일반사항 > 1.1 적용범위",
                "page": scope_page,
                "content": scope_content,
                "search_text": f"{code} {scope_content}",
            },
            {
                "hierarchy": (
                    f"[{code}] 4 검사기준 > 4.2 검사방법 > 4.2.1 중간검사 > "
                    "4.2.1.5 내압 및 기밀 시험방법 > 4.2.1.5.2 기밀시험방법"
                ),
                "page": test_page,
                "content": clause_content,
                "search_text": f"{code} 기밀시험 시험압력 기밀유지시간 합격",
            },
        ]
        if code == "FP217":
            rows.append(
                {
                    "hierarchy": f"[{code}] 4.2.2.3 검지경보장치",
                    "page": table_page,
                    "content": (
                        "압력측정기구 용적 기밀유지시간 압력계 또는 자기압력기록계 "
                        "1m3 미만 48분 1m3 이상 10m3 미만 480분 10m3 이상 48×V분 "
                        "(다만, 2880분을 초과한 경우는 2 880분으로 할 수 있다) "
                        "[비고] V는 피시험부분의 용적(단위 : m3)이다. "
                        "충전시설에 설치된 검지경보장치의 설치 여부를 확인한다."
                    ),
                    "search_text": f"{code} 기밀시험 기밀유지시간 압력측정기구 용적",
                }
            )
        database.replace_document(
            {
                "doc_type": "CODE",
                "doc_code": code,
                "title": f"{code} 수소연료 충전시설 기준",
                "filename": f"{code}.pdf",
                "file_path": str(tmp_path / f"{code}.pdf"),
                "file_hash": f"{code}-tightness-detailed-comparison",
                "page_count": 120,
            },
            rows,
        )

    pipeline = RagPipeline(
        SimpleNamespace(
            service_hub_model="test",
            service_hub_fast_model="test-fast",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        database,
        ReasonerMustNotRun(),
    )
    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FP216과 FP217의 배관 기밀시험을 비교해줘. 시험매체, 시험압력, "
                "실제 시험구간 용적 12m³에서의 유지시간, 합격판정을 항목별로 나누고 각 기준의 적용 대상과 "
                "조항·PDF 쪽수를 함께 보여줘. 원문에 없는 조건은 추정하지 마."
            )
        )
    )

    assert response.intent == "comparison"
    assert "제조식 시설" in response.answer and "저장식 시설" in response.answer
    assert "공기 또는 위험성이 없는 기체" in response.answer
    assert "0.7MPa를 초과하는 경우 0.7MPa압력 이상으로 한다" in response.answer
    assert "1 m³ 미만 48분" in response.answer
    assert "1 m³ 이상 10 m³ 미만 480분" in response.answer
    assert "10 m³ 이상 48×V분" in response.answer
    assert "48×12=576분" in response.answer
    assert "처음과 마지막" in response.answer and "누설 등의 이상이 없을 때 합격" in response.answer
    assert len(response.citations) == 10
    assert [item.page for item in response.citations] == [13, 12, 111, 111, 111, 111, 95, 95, 95, 96]
    assert response.citations[5].excerpt.startswith("표 4.2.1.5.2")
    assert response.citations[-1].hierarchy.endswith("표 시험 용적에 따른 기밀유지시간")
    assert "검지경보장치" not in response.citations[-1].excerpt


@pytest.mark.asyncio
async def test_fs551_fu671_hydrogen_gas_comparison_keeps_distinct_conditions():
    fs551 = {
        "document_id": 10,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "도시가스 배관 기준",
        "chunk_id": 1001,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.9.3 배관의 기밀시험 방법은 다음과 같다. "
            "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. 다만, 통과하는 가스로 "
            "기밀시험을 할 수 있는 경우는 다음과 같다. "
            "(1-1) 최고사용압력이 고압이나 중압으로 길이가 15m 미만인 배관으로서 이음부와 "
            "동일재료, 동일치수 및 동일시공방법에 따르고 최고사용압력의 1.1배 이상에서 누출이 없고 "
            "4.2.2.9.4(1) 또는 (2) 방법으로 시험한 경우 "
            "(1-2) 최고사용압력이 저압인 배관으로서 4.2.2.9.4(1) 또는 (2) 방법으로 시험한 경우 "
            "(1-3) 기설치된 사용자공급관의 기밀시험을 하는 경우 (2) 시험압력 기준이다. "
            "4.2.2.9.4 신규로 설치되는 본관, 공급관의 기밀시험은 다음 방법으로 실시한다. "
            "(3) 최고사용압력이 고압이나 중압인 배관으로서 용접 접합되고 방사선투과시험에 따라 "
            "합격된 배관은 통과하는 가스를 시험가스로 사용하고 0.2% 이하에서 작동하는 가스검지기를 "
            "사용하여 해당 검지기가 작동하지 않는 것으로 판정한다(매설된 배관은 시험가스를 넣어 "
            "24시간 경과한 후 판정한다). 이때 시험압력은 사용압력으로 할 수 있다."
        ),
    }
    fu671 = {
        "document_id": 11,
        "doc_type": "CODE",
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1002,
        "hierarchy": "[FU671] 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "filename": "FU671.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 수소를 사용하여 기밀시험을 할 수 있다. "
            "이 경우 압력은 단계적으로 올려 이상이 없음을 확인하면서 승압한다. (5) 합격 기준이다."
        ),
    }

    class FS551FU671ComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code in {"FS551", "FU671"}}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert set(codes) == {"FS551", "FU671"}
            assert limit == 100
            return [fs551, fu671]

        def save_exchange(self, *args):
            return 8

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FS551FU671ComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message=(
                "KGS FS551과 FU671 기밀시험에서 수소를 시험가스로 쓸 수 있는 조건을 비교해줘. "
                "FS551의 통과가스와 FU671의 수소 예외를 다른 규칙으로 구분해줘."
            )
        )
    )

    assert response.intent == "comparison"
    assert "두 기준의 시험가스 예외는 같은 규칙이 아닙니다" in response.answer
    assert "FS551 4.2.2.9.3(1)" in response.answer
    assert "15m 미만" in response.answer and "기설치된 사용자공급관" in response.answer
    assert "신규 본관·공급관의 별도 방법" in response.answer
    assert "방사선투과시험" in response.answer and "24시간" in response.answer
    assert "0.2% 이하" in response.answer and "작동하지 않는 것으로 판정" in response.answer
    assert "FU671 4.2.2.9.3(4)" in response.answer
    assert "위험이 없다고 판단" in response.answer and "단계적으로 올리면서" in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FS551", "FU671"]
    assert [citation.page for citation in response.citations] == [94, 95]
    assert "4.2.2.9.4(3)" in response.citations[0].hierarchy
    assert "수소를 사용하여 기밀시험" in response.citations[1].excerpt


def test_fs551_fu671_detail_comparison_keeps_media_and_hold_time_rules_separate():
    fs_clause = {
        "document_id": 10,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "도시가스 배관 기준",
        "chunk_id": 1010,
        "hierarchy": "[FS551] 4.2.2.9.3 배관의 기밀시험 방법",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.9.3 (1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. "
            "다만, 통과하는 가스로 기밀시험을 할 수 있는 경우는 다음과 같다. "
            "(1-1) 최고사용압력이 고압이나 중압으로 길이가 15m 미만인 배관. "
            "(1-2) 최고사용압력이 저압인 배관. (1-3) 기설치된 사용자공급관. "
            "(2) 시험압력 기준"
        ),
    }
    fs_method = {
        **fs_clause,
        "chunk_id": 1011,
        "hierarchy": "[FS551] 4.2.2.9.4 신규 배관 기밀시험 방법",
        "page": 95,
        "content": (
            "4.2.2.9.4 신규로 설치되는 본관, 공급관의 기밀시험 방법. "
            "(2) 가스검지기 방법은 매설배관에서 시험가스를 넣고 12시간 경과 후 판정한다. "
            "(3) 용접·방사선투과시험 합격 배관은 통과하는 가스를 시험가스로 사용하고, "
            "매설된 배관은 시험가스를 넣어 24시간 경과 후 판정한다. "
            "(4) 압력측정기구 방법은 시험부 용적에 따른 기밀유지시간을 유지한다."
        ),
    }
    fu_hold = {
        **fs_clause,
        "doc_code": "FU671",
        "title": "수소연료사용시설 기준",
        "chunk_id": 1012,
        "hierarchy": "[FU671] 4.2.2.9.3 기밀시험방법",
        "page": 95,
        "content": (
            "(1) 기밀시험은 원칙적으로 공기 또는 위험성이 없는 기체의 압력으로 실시한다. "
            "(4) 위험이 없다고 판단되는 경우 수소를 사용하여 기밀시험을 할 수 있고 압력을 단계적으로 올린다. "
            "표 4.2.2.9.3 시험 용적에 따른 기밀유지시간. 1㎥ 미만 24분, "
            "1㎥ 이상 10㎥ 미만 240분, 10㎥ 이상 24×V분. 다만 1440분을 초과하면 1440분으로 할 수 있다."
        ),
    }
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        SimpleNamespace(),
        ReasonerMustNotRun(),
    )
    result = pipeline._fs551_fu671_tightness_detail_comparison(
        "KGS FS551과 FU671의 기밀시험에서 시험매체, 통과가스 허용 조건, 기밀유지시간을 항목별로 비교해줘.",
        [fs_clause, fs_method, fu_hold],
    )
    assert result is not None
    answer, citations = result
    assert "FS551" in answer and "FU671" in answer
    assert "12시간" in answer and "24시간" in answer
    assert "1㎥ 미만 24분" in answer and "24×V분" in answer
    assert [citation.doc_code for citation in citations] == ["FS551", "FS551", "FU671"]
    assert [citation.page for citation in citations] == [94, 95, 95]


@pytest.mark.asyncio
async def test_fp217_ambiguous_tightness_pressure_clause_is_quoted_without_extrapolation():
    chunk = {
        "document_id": 3,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 993,
        "hierarchy": "[FP217] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "기밀시험은 다음 기준으로 실시한다. "
            "(3) 기밀시험압력은 상용압력 이상으로 하되, 0.7MPa를 초과하는 경우 "
            "0.7MPa압력 이상으로 한다. (4) 시험 시 안전을 확인한다."
        ),
    }

    class FP217PressureDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {"FP217"} if "FP217" in codes else set()

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 3

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FP217PressureDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message="KGS FP217 기밀시험 압력 기준에서 0.7 MPa 조건은?"))

    assert response.intent == "fact"
    assert "상용압력 이상으로 하되, 0.7MPa를 초과하는 경우 0.7MPa압력 이상" in response.answer
    assert "문장만으로는 명료하지 않습니다" in response.answer
    assert "유지시간" not in response.answer and "표 4.2.1.5.2" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].doc_code == "FP217"
    assert response.citations[0].page == 95
    assert response.citations[0].excerpt == (
        "기밀시험압력은 상용압력 이상으로 하되, 0.7MPa를 초과하는 경우 0.7MPa압력 이상으로 한다."
    )


@pytest.mark.asyncio
async def test_fp217_stored_gas_exception_answers_only_the_requested_condition_and_ramp():
    chunk = {
        "document_id": 4,
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "저장식 수소연료 충전 기준",
        "chunk_id": 994,
        "hierarchy": "[FP217] 4 검사기준 > 4.2.1.5.2 기밀시험방법",
        "page": 95,
        "filename": "FP217.pdf",
        "score": 1.0,
        "content": (
            "기밀시험방법. (1) 기밀시험은 공기로 실시한다. "
            "(3) 시험압력은 상용압력 이상으로 한다. "
            "(4) 검사의 상황에 따라 위험이 없다고 판단되는 경우에는 해당 고압가스설비에 "
            "저장 또는 처리되는 가스를 사용하여 기밀시험을 할 수 있다. 이 경우 압력은 "
            "단계적으로 올려 이상이 없는지를 확인하면서 승압한다. "
            "(5) 기밀시험압력에서 누설 등의 이상이 없을 때 합격으로 한다."
        ),
    }

    class FP217StoredGasDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {"FP217"} if "FP217" in codes else set()

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FP217"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 4

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FP217StoredGasDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="KGS FP217에서 저장 또는 처리되는 가스로 기밀시험할 수 있는 요건과 압력 올리는 방법은?"
        )
    )

    assert "위험이 없다고 판단되는 경우" in response.answer
    assert "해당 고압가스설비에 저장 또는 처리되는 가스" in response.answer
    assert "압력은 단계적으로 올리면서 이상이 없는지 확인" in response.answer
    assert "0.7MPa" not in response.answer
    assert "상용압력" not in response.answer and "누설 등의 이상" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 96
    assert "(4)" not in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fu671_pressure_terms_are_compared_and_not_treated_as_interchangeable(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-pressure-test",
            "page_count": 15,
        },
        [
            {
                "hierarchy": "[FU671] 1 일반사항 > 1.3 용어 정의 > 1.3.6.2 제2종보호시설",
                "page": 12,
                "content": (
                    "1.3.8 “상용압력”이란 내압시험압력 및 기밀시험압력의 기준이 되는 압력으로서 "
                    "사용상태에서 해당 설비 등의 각부에 작용하는 최고사용압력을 말한다. "
                    "1.3.10 “설정압력(set pressure)”이란 안전밸브의 설계상 정한 분출압력 또는 "
                    "분출개시압력으로서 명판에 표시된 압력을 말한다."
                ),
                "search_text": "FU671 상용압력 설정압력 내압시험 기밀시험 안전밸브",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671의 정의 조항 기준으로 상용압력과 설정압력의 차이를 각각 설명해줘. "
                "둘을 서로 바꾸어 써도 되는지도 근거에 따라 답해줘."
            )
        )
    )

    assert "내압시험압력·기밀시험압력의 기준" in response.answer
    assert "안전밸브" in response.answer and "명판에 표시" in response.answer
    assert "같은 뜻이 아닙니다" in response.answer
    assert "성능을 평가" not in response.answer
    assert len(response.citations) == 1
    assert "1.3.8 상용압력 및 1.3.10 설정압력" in response.citations[0].hierarchy
    assert "1.3.8 상용압력이란" in response.citations[0].excerpt
    assert "1.3.10 설정압력이란" in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fu671_indoor_detector_count_labels_rounding_as_arithmetic_inference(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-count-test",
            "page_count": 80,
        },
        [
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 가스누출경보기 "
                    "> 2.8.2.3 설치장소 및 설치개수 > 2.8.2.3.1 사업소 안"
                ),
                "page": 70,
                "content": (
                    "(1) 건축물 안에 설치된 수소가스설비군은 바닥면 둘레 10m마다 "
                    "1개 이상의 비율로 계산한 수 (2) 건축물 밖 설비는 다른 가스설비에 인접한 경우 "
                    "둘레 20m마다 1개 이상의 비율로 계산한 수"
                ),
                "search_text": "FU671 건축물 안 바닥면 둘레 10m마다 검지부 1개",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 2.8.2.3.1(1)에서 건축물 안 설비군 바닥면 둘레가 25m라면 "
                "검지부 최소 개수는 몇 개로 계산돼? 10m당 1개 비율의 산식을 보여주고, "
                "올림 규정이 원문에 따로 있는지와 정수 개수로 계산한 추론을 구분해줘."
            )
        )
    )

    assert "25m ÷ 10m × 1개 = 2.5개" in response.answer
    assert "산술상 최소 3개" in response.answer
    assert "별도의 올림 절차를 적지는 않습니다" in response.answer
    assert response.citations[0].page == 70

    plain_language_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 건물 안 수소 생산설비군 바닥 둘레가 25m이면 "
                "가스누출검지기 최소 몇 개 배치해야 해? 산식을 보여줘."
            )
        )
    )
    assert "25m ÷ 10m × 1개 = 2.5개" in plain_language_response.answer
    assert "산술상 최소 3개" in plain_language_response.answer
    assert plain_language_response.citations[0].page == 70


def test_fu671_indoor_detector_count_handles_exact_interval_without_rounding():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU671",
        "hierarchy": "[FU671] 2.8.2.3.1 사업소 안",
        "content": "(1) 건축물 안 설비군의 바닥면 둘레 10m마다 1개 이상의 비율로 계산한 수 (2) 밖",
    }

    result = RagPipeline._fu671_detector_count_for_perimeter(
        "KGS FU671 2.8.2.3.1(1)에서 건축물 안 설비군 바닥면 둘레가 정확히 30m이면 최소 몇 개야?",
        [chunk],
    )

    assert result is not None
    _source, answer, _excerpt = result
    assert "30m ÷ 10m × 1개 = 3개" in answer
    assert "따라서 최소 3개입니다" in answer
    assert "별도의 올림 절차" not in answer


@pytest.mark.asyncio
async def test_fu671_high_ceiling_detector_location_keeps_distance_and_hood_dimensions(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-clearance-test",
            "page_count": 80,
        },
        [
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 경보기 > "
                    "2.8.2.3 설치장소 및 설치개수 > 2.8.2.3.1 사업소 안"
                ),
                "page": 70,
                "content": (
                    "(1) 건축물 안 수소가스설비군 주위의 누출가스 체류 우려 장소에 "
                    "설비군 바닥면 둘레 10m마다 1개 이상으로 계산한다. "
                    "(2) 건축물 밖에서 다른 설비·구조물에 인접하거나 가스가 체류할 수 있는 설비는 "
                    "설비군 바닥면 둘레 20m마다 1개 이상으로 계산한다. (3) 가열로 등 발화원은 "
                    "체류 우려 장소의 바닥면 둘레 20m마다 1개 이상으로 계산한다."
                ),
                "search_text": "FU671 수소가스설비군 누출 체류 바닥면 둘레 10m 20m 검지부 개수",
            },
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 경보기 > "
                    "2.8.2.3 설치장소 및 설치개수 > 2.8.2.3.2 사업소 밖"
                ),
                "page": 70,
                "content": (
                    "2.8.2.3.2 사업소 밖: (1) 긴급차단 장치가 설치된 부분 "
                    "(2) 밀폐되어 설치되는 부분 (3) 누출 가스가 체류하기 쉬운 구조인 부분. "
                    "2.8.2.3.3 검지경보장치의 검출부 설치 위치는 천정으로부터 검지부 하단까지의 거리가 "
                    "0.3m 이하가 되도록 설치한다. 2.8.2.3.4 2.8.2.3.3에도 불구하고 공장 등과 같이 "
                    "천장높이가 지나치게 높은 건물에서 검지경보장치 검출부를 천장부분에 설치할 경우에는 "
                    "다량의 가스누출이 되어 위험한 상태가 되어야만 검지가 가능하므로 이를 보완하기 위하여 "
                    "다음과 같이 포집갓을 설치한다. (1) 가스가 소량누출시 검지가 가능하도록 수소가스설비 중 "
                    "가스가 누출되기 쉬운 부분의 상부에 검출부를 설치하고 가스 누출 시 포집이 가능하도록 "
                    "검출부에 포집갓을 설치한다. (2) 포집갓의 규격은 가로, 세로 0.4m 이상(사각형의 경우) "
                    "또는 직경 0.4m 이상(원형의 경우)이 되도록 한다. 2.8.2.3.5 경보부 설치 장소"
                ),
                "search_text": "FU671 천정 검지부 0.3m 높은 건물 포집갓 0.4m",
            },
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 경보기 > "
                    "2.8.2.1 가스누출경보기 및 가스누출자동차단장치 기능"
                ),
                "page": 68,
                "content": (
                    "2.8.2.1.2 경보농도는 검지경보장치의 설치장소, 주위 분위기 온도에 따라 "
                    "폭발 하한계의 1/4 이하 이하로 한다."
                ),
                "search_text": "FU671 경보기 설정값 경보농도 폭발하한계 1/4",
            },
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 천장 높이가 매우 높은 공장에서는 누출감지기를 어디에 설치하고, "
                "포집후드(포집갓) 크기는 얼마로 해야 해?"
            )
        )
    )

    assert "검지부 하단까지의 거리를 0.3m 이하" in response.answer
    assert "누출되기 쉬운 설비 부분의 위쪽" in response.answer
    assert "가로·세로 각각 0.4m 이상" in response.answer
    assert "지름 0.4m 이상" in response.answer
    assert response.citations[0].page == 70
    assert "2.8.2.3.4" in response.citations[0].excerpt
    assert "2.8.2.3.3–2.8.2.3.4" in response.citations[0].hierarchy

    combined_response = await pipeline.run(
        ChatRequest(
            message="KGS FU671 검지경보장치 검출부 설치 위치와 경보기 설정값은?"
        )
    )
    assert "0.3m 이하" in combined_response.answer
    assert "폭발하한계(LEL)의 1/4 이하" in combined_response.answer
    assert "이하 이하" not in combined_response.answer
    assert [citation.page for citation in combined_response.citations] == [70, 69]
    assert [citation.number for citation in combined_response.citations] == [1, 2]
    assert "이하 이하" not in combined_response.citations[1].excerpt

    heavy_gas_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 검지부는 천장과 바닥에서 각각 어느 거리야? "
                "공기보다 무거운 가스의 바닥 설치 예외도 분리해줘."
            )
        )
    )
    assert "천정에서 검지부 하단까지의 거리를 0.3m 이하" in heavy_gas_response.answer
    assert "바닥면부터 검지부 상단까지 0.3m 이하로 하라는 예외는 이 FU671 조항에 없습니다" in heavy_gas_response.answer
    assert len(heavy_gas_response.citations) == 1
    assert heavy_gas_response.citations[0].page == 70

    source_distance_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 수소 가스누출검지기 설치 위치와 누출원과의 거리 제한은? "
                "무거운 가스는 바닥에서 몇 cm 기준인지도 알려줘."
            )
        )
    )
    assert "천정에서 검지부 하단까지의 거리" in source_distance_response.answer
    assert "누출원과의 고정된 최대 이격거리는 이 조항에 제시되지 않습니다" in source_distance_response.answer
    assert "10m마다" in source_distance_response.answer
    assert "20m마다" in source_distance_response.answer
    assert "둘레당 검지부 수량 비율" in source_distance_response.answer
    assert "바닥면부터 검지부 상단까지 0.3m 이하로 하라는 예외는 이 FU671 조항에 없습니다" in source_distance_response.answer
    assert [citation.number for citation in source_distance_response.citations] == [1, 2]
    assert [citation.page for citation in source_distance_response.citations] == [70, 70]


@pytest.mark.asyncio
async def test_fu671_detector_prohibition_question_reports_clause_scope_instead_of_abstaining(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-detector-prohibition-test",
            "page_count": 80,
        },
        [
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > "
                    "2.8.2.3 가스누출경보기 및 가스누출자동차단장치 설치장소 및 설치개수 > "
                    "2.8.2.3.2 사업소 밖"
                ),
                "page": 70,
                "content": (
                    "검지경보장치의 검출부 설치장소 및 설치 개수는 다음 기준에 따른다. "
                    "2.8.2.3.2 사업소 밖: (1) 긴급차단 장치가 설치된 부분 "
                    "(2) 밀폐되어 설치되는 부분 (3) 누출 가스가 체류하기 쉬운 구조인 부분. "
                    "2.8.2.3.3 검지경보장치의 검출부 설치 위치는 천정으로부터 검지부 하단까지 "
                    "0.3m 이하가 되도록 설치한다."
                ),
                "search_text": "FU671 검지경보장치 설치장소 사업소 밖 검출부 천정",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(message="KGS FU671 가스누출경보기는 어떤 곳에 설치하면 안 돼?")
    )

    assert "목록이 열거되어 있지 않습니다" in response.answer
    assert "특정 장소를 설치 금지라고 단정할 수 없습니다" in response.answer
    assert response.intent == "fact"
    assert len(response.citations) == 1
    assert response.citations[0].page == 70
    assert "2.8.2.3.2" in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fu671_alarm_time_qualifier_is_not_rewritten_as_an_absolute_cap(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-time-test",
            "page_count": 80,
        },
        [
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 경보기 > "
                    "2.8.2.1 가스누출경보기 및 가스누출자동차단장치 기능"
                ),
                "page": 68,
                "content": (
                    "2.8.2.1.4 검지에서 발신까지 걸리는 시간은 경보농도의 1.6배 농도에서 "
                    "보통 30초 이내로 한다. 2.8.2.1.5 전원 전압이 변동해도 정밀도가 저하되지 않아야 한다."
                ),
                "search_text": "FU671 경보농도 1.6배 보통 30초 검지 발신",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "FU671 2.8.2.1.4의 ‘보통 30초 이내’는 무조건적인 절대 상한이라고 볼 수 있어? "
                "해당 조항의 문언과 그로부터 추론할 수 있는 범위를 나눠 답해줘."
            )
        )
    )

    assert "‘보통 30초 이내’" in response.answer
    assert "절대 상한이라고 단정하는 것은 원문보다 강한 표현" in response.answer
    assert "예외나 30초 초과 허용 조건도 따로 적혀 있지 않아" in response.answer
    assert response.citations[0].page == 69
    assert "1.6배 농도에서 보통 30초 이내" in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fu671_alarm_threshold_and_signal_time_are_extracted_together(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-alarm-parameters-test",
            "page_count": 80,
        },
        [
            {
                "hierarchy": (
                    "[FU671] 2 시설기준 > 2.8 사고예방설비기준 > 2.8.2 경보기 > "
                    "2.8.2.1 가스누출경보기 및 가스누출자동차단장치 기능"
                ),
                "page": 68,
                "content": (
                    "2.8.2.1.2 경보농도는 설치장소 및 주위의 분위기 온도에 따라 "
                    "수소 폭발 하한계의 1/4 이하 이하로 한다. "
                    "2.8.2.1.3 정밀도는 설정 경보농도의 ±25% 이하로 한다. "
                    "2.8.2.1.4 검지에서 발신까지 걸리는 시간은 경보농도의 1.6배 농도에서 "
                    "보통 30초 이내로 한다."
                ),
                "search_text": "FU671 경보농도 설정 폭발하한계 1/4 정밀도 ±25% 1.6배 발신 30초",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    source_chunk = database.search_headings(
        "가스누출경보기 및 가스누출자동차단장치 기능", ["FU671"], limit=100
    )[0]
    assert pipeline.citations([source_chunk])[0].page == 69

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 가스누출경보기의 경보농도 설정 기준과 감지 후 신호 발신까지의 "
                "시간 기준을 알려줘."
            )
        )
    )

    assert "폭발하한계(LEL)의 1/4 이하" in response.answer
    assert "1.6배 농도에서 검지부터 발신까지 보통 30초 이내" in response.answer
    assert "이하 이하" not in response.citations[0].excerpt
    assert response.citations[0].page == 69
    assert "2.8.2.1.2–2.8.2.1.4" in response.citations[0].hierarchy

    threshold_precision_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 경보농도는 폭발하한계의 몇 분의 몇이고, "
                "경보기 정밀도 허용범위는 얼마야?"
            )
        )
    )
    assert "폭발하한계(LEL)의 1/4 이하" in threshold_precision_response.answer
    assert "설정 경보농도의 ±25% 이하" in threshold_precision_response.answer
    assert "30초" not in threshold_precision_response.answer
    assert "2.8.2.1.2–2.8.2.1.3" in threshold_precision_response.citations[0].hierarchy
    assert threshold_precision_response.citations[0].page == 69
    assert "2.8.2.1.4" not in threshold_precision_response.citations[0].excerpt

    numeric_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671에서 경보기 설정 농도는 폭발하한계의 1/4 이하야. "
                "해당 가스의 LEL이 4.0 vol%라면 설정 가능한 최대 농도는 얼마야? "
                "측정값 0.8 vol%가 자동으로 경보 대상인지도 구분해서 알려줘."
            )
        )
    )
    assert "LEL 4.0 vol%의 1/4은 1.0 vol%" in numeric_response.answer
    assert "설정 상한은 1.0 vol%" in numeric_response.answer
    assert "0.8 vol%가 자동으로 경보되는지는 실제 설정 경보농도에 따라 달라집니다" in numeric_response.answer
    assert "실제 장치의 설정값을 확인해야 합니다" in numeric_response.answer
    assert numeric_response.citations[0].page == 69

    configured_alarm_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671에서 수소 LEL이 4.0 vol%이고 경보기 설정값이 0.8 vol%야. "
                "이 농도에서 경보가 반드시 울려야 해? 기준의 시험 농도와 발신시간도 구분해줘."
            )
        )
    )
    assert "설정값 0.8 vol%는 계산상 상한 1.0 vol% 이하" in configured_alarm_response.answer
    assert "설정 경보농도 0.8 vol%의 1.6배는 1.28 vol%" in configured_alarm_response.answer
    assert "정확히 설정농도에 도달한 순간의 즉시 발신을 이 문언만으로 보장한다고 단정할 수 없습니다" in configured_alarm_response.answer
    assert "2.8.2.1.4" in configured_alarm_response.citations[0].hierarchy

    over_limit_alarm_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671에서 수소의 LEL이 4.0 vol%이고 경보기 설정값을 1.2 vol%로 두어도 "
                "기준상 허용돼? 계산 근거와 설치환경 조건을 설명해줘."
            )
        )
    )
    assert "LEL 4.0 vol%의 1/4은 1.0 vol%" in over_limit_alarm_response.answer
    assert "설정값 1.2 vol%는 계산상 상한 1.0 vol%를 넘습니다" in over_limit_alarm_response.answer
    assert "기준에 맞지 않습니다" in over_limit_alarm_response.answer

    actual_point_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671에서 폭발하한계가 4%인 가스의 경보농도를 1.2%로 설정하면 기준 이내야? "
                "허용 경보농도 상한과 실제 경보 작동 시점은 구분해서 설명해줘."
            )
        )
    )
    assert "LEL 4 vol%의 1/4은 1 vol%" in actual_point_response.answer
    assert "설정값 1.2 vol%는 계산상 상한 1 vol%를 넘습니다" in actual_point_response.answer
    assert "정밀도 ±25%" in actual_point_response.answer
    assert "실제 작동 농도는 장치의 설정 경보농도와 정밀도에 좌우됩니다" in actual_point_response.answer
    assert "정밀도는 설정치 주변의 오차 한도이지 설정농도의 허용 상한" in actual_point_response.answer
    assert "설정 경보농도 1.2 vol%의 1.6배는 1.92 vol%" in actual_point_response.answer
    assert "이 농도에서 검지부터 발신까지는 보통 30초 이내" in actual_point_response.answer
    assert "2.8.2.1.4" in actual_point_response.citations[0].excerpt
    assert actual_point_response.citations[0].page == 69

    signal_calculation_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 경보기 설정농도가 1.0 vol%라면 그 1.6배 농도는 얼마이고, "
                "그 농도에서 신호 발신시간 기준은 어떻게 돼?"
            )
        )
    )
    assert "설정 경보농도 1.0 vol%의 1.6배는 1.6 vol%" in signal_calculation_response.answer
    assert "이 농도에서 검지부터 발신까지는 보통 30초 이내" in signal_calculation_response.answer
    assert "2.8.2.1.4" in signal_calculation_response.citations[0].excerpt

    small_lel_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 경보농도 기준에서 LEL이 0.05 vol%라면 설정 상한은 얼마야?"
            )
        )
    )
    assert "0.05 vol%의 1/4은 0.0125 vol%" in small_lel_response.answer

    small_setpoint_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 경보기 설정농도가 0.03 vol%라면 1.6배 농도와 "
                "그 농도에서의 신호 발신시간은?"
            )
        )
    )
    assert "0.03 vol%의 1.6배는 0.048 vol%" in small_setpoint_response.answer

    precision_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671에서 설정 경보농도가 0.8 vol%라면 정밀도 ±25%에 해당하는 "
                "농도 범위는 얼마야? 이 정밀도를 경보기 설정 허용 상한으로 봐도 돼?"
            )
        )
    )
    assert "최대 ±0.2 vol%" in precision_response.answer
    assert "단순 환산 범위는 0.6–1.0 vol%" in precision_response.answer
    assert "설정농도의 허용 상한을 뜻하지 않습니다" in precision_response.answer
    assert "0.8 vol% 설정 자체가 허용되는지는 LEL의 1/4 기준" in precision_response.answer

    fault_diagnosis_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 수소 누출 상황에서 LEL=4.0 vol%이고 경보기 표시가 0.8 vol%인데 "
                "경보음이 안 울렸어. 이 사실만으로 오작동이라고 결론낼 수 있어? "
                "기준상 어떤 설정값과 시간 정보를 확인해야 하는지 설명해줘."
            )
        )
    )
    assert "LEL 4.0 vol%의 1/4은 1.0 vol%" in fault_diagnosis_response.answer
    assert "표시농도 0.8 vol%만으로는 오작동이라고 단정할 수 없습니다" in fault_diagnosis_response.answer
    assert "실제 설정값이 0.8 vol%보다 높으면 아직 설정농도 미만" in fault_diagnosis_response.answer
    assert "경보농도의 1.6배 농도에서 검지부터 발신까지 보통 30초 이내" in fault_diagnosis_response.answer
    assert "2.8.2.1.4" in fault_diagnosis_response.citations[0].excerpt


def test_low_pressure_scope_limit_does_not_infer_pressure_type():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "AA012",
        "content": "이 기준은 저압 (3.3 kPa 이하) 전용 가스시설에 적용한다.",
    }

    assert RagPipeline._code_scope_low_pressure_limit(
        "KGS AA012에서 3.3 kPa는 최고사용압력이야, 조정기 설정압력이야?", [chunk]
    ) is None


def test_grounded_claim_validator_rejects_changed_frequency():
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
        title="배관 기준", hierarchy="점검", page=1, filename="FS551.pdf",
        excerpt="일반도시가스사업자는 6월에 1회 이상 긴급차단장치를 점검한다.", score=1.0,
    )
    source = [{"content": "일반도시가스사업자는 6월에 1회 이상 긴급차단장치를 점검한다."}]
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(
                claim="긴급차단장치는 연 1회 점검한다.", citation_number=1,
                exact_quote="6월에 1회 이상 긴급차단장치를 점검한다.",
            ),
            EvidenceClaim(
                claim="긴급차단장치는 매년 6월에 1회 이상 점검한다.", citation_number=1,
                exact_quote="6월에 1회 이상 긴급차단장치를 점검한다.",
            ),
            EvidenceClaim(
                claim="긴급차단장치는 6월에 1회 이상 점검한다.", citation_number=1,
                exact_quote="6월에 1회 이상 긴급차단장치를 점검한다.",
            ),
        ],
        insufficient_evidence=False,
    )

    answer, used = RagPipeline._validated_claim_answer(grounded, [citation], source)

    assert "연 1회" not in answer
    assert "매년" not in answer
    assert "6월에 1회 이상" in answer
    assert len(used) == 1
    assert used[0].excerpt == "6월에 1회 이상 긴급차단장치를 점검한다."


def test_quantitative_claim_renders_source_wording_to_preserve_conditions():
    source_text = (
        "중압 이상의 배관은 최고사용압력의 1.5배(고압의 가스시설로서 "
        "공기·질소 등의 기체로 실시하는 경우에는 1.25배) 이상의 압력으로 시험한다."
    )
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
        title="배관 기준", hierarchy="내압시험", page=98, filename="FS551.pdf",
        excerpt=source_text, score=1.0,
    )
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(
                claim="중압은 1.5배, 고압은 1.25배로 시험한다.",
                citation_number=1,
                exact_quote=source_text,
            )
        ],
        insufficient_evidence=False,
    )

    answer, used = RagPipeline._validated_claim_answer(
        grounded, [citation], [{"content": source_text}], numbered=True
    )

    assert "공기·질소 등의 기체로 실시하는 경우" in answer
    assert used == [citation]


def test_verified_quote_repairs_known_pdf_word_splits_in_answer_text():
    source_text = "폭발하한계의 1/4 이하에서 60초 이내에 경보를 울리는 것으로 한 다."
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FU551",
        title="가스사용시설 기준", hierarchy="가스누출경보기", page=71,
        filename="FU551.pdf", excerpt=source_text, score=1.0,
    )
    grounded = GroundedClaims(
        claims=[EvidenceClaim(
            claim="폭발하한계의 1/4 이하에서 60초 이내에 경보한다.",
            citation_number=1,
            exact_quote=source_text,
        )],
        insufficient_evidence=False,
    )

    answer, _used = RagPipeline._validated_claim_answer(
        grounded, [citation], [{"content": source_text}]
    )

    assert "한 다" not in answer
    assert "한 다." not in answer


def test_nominal_pipe_diameter_cannot_gain_an_unsupported_exclusion():
    source_text = "최고사용압력이 저압으로서 호칭지름 50A 이상의 노출 배관"
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
        title="배관 기준", hierarchy="배관 접합", page=34,
        filename="FS551.pdf", excerpt=source_text, score=1.0,
    )
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(
                claim="저압 50A 이상 노출배관은 비파괴시험 대상에서 제외된다.",
                citation_number=1,
                exact_quote=source_text,
            )
        ],
        insufficient_evidence=False,
    )

    with pytest.raises(ValueError, match="검증을 통과한 주장이 부족합니다"):
        RagPipeline._validated_claim_answer(
            grounded, [citation], [{"content": source_text}]
        )


def test_conditional_claim_renders_full_source_to_preserve_certificate_requirements():
    source_text = (
        "제품번호(Lot No 등)가 관리되고, 각 제품에 대하여 전문 비파괴검사업소에서 발행한 "
        "비파괴시험 성적서가 있는 경우에는 그 시험성적서를 비파괴시험에 갈음하여 현장에서 "
        "별도의 비파괴시험을 하지 않을 수 있다."
    )
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
        title="배관 기준", hierarchy="접합", page=92, filename="FS551.pdf", excerpt=source_text, score=1.0,
    )
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(
                claim="성적서가 있으면 현장시험을 생략할 수 있다.",
                citation_number=1,
                exact_quote=source_text,
            )
        ],
        insufficient_evidence=False,
    )

    answer, used = RagPipeline._validated_claim_answer(
        grounded, [citation], [{"content": source_text}], numbered=True
    )

    assert "각 제품에 대하여 전문 비파괴검사업소에서 발행한" in answer
    assert "하지 않을 수 있다" in answer
    assert used == [citation]


def test_procedure_answer_requires_every_requested_section():
    citation = Citation(
        number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
        title="배관 기준", hierarchy="[FS551] 4 검사 기준 > 4.2.2.1 설치상황",
        page=91, filename="FS551.pdf", excerpt="배관 위치를 확인한다.", score=1.0,
    )
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(
                claim="배관 위치를 확인한다.", citation_number=1,
                exact_quote="배관 위치를 확인한다.",
            )
        ],
        insufficient_evidence=False,
    )

    with pytest.raises(ValueError, match="필수 절 근거"):
        RagPipeline._validated_claim_answer(
            grounded,
            [citation],
            [{"content": "배관 위치를 확인한다."}],
            numbered=True,
            minimum_claims=2,
            required_headings=("4.2.2.1", "4.2.2.2"),
        )


@pytest.mark.asyncio
async def test_fs551_main_inspection_procedure_uses_complete_cited_clause_order_without_llm():
    clauses = [
        ("4.1.2 시공감리", "배관에 대한 시공감리 항목은 용품·압력·설치 제한 등을 확인한다."),
        ("4.1.3 정기검사", "배관에 대한 정기검사는 특정 사용자공급관을 제외한 배관에만 실시한다. (1) 항목"),
        ("4.1.4 수시검사", "배관에 대한 수시검사 항목은 4.1.3의 정기검사 항목을 따른다."),
        ("4.2 검사방법", "검사는 대상시설이 시설기준 및 기술기준에 적합한지 판정하도록 실시한다."),
        ("4.2.2 시공감리, 정기검사 및 수시검사", "검사방법은 다음과 같다. 다만, 정기검사 및 수시검사 시에는 4.2.2.1부터 4.2.2.3까지 및 4.2.2.10을 제외할 수 있다."),
        ("4.2.2.1 설치상황", "4.2.2.1.1 배관부설위치와 심도가 공사계획에 적정한지 확인한다. 4.2.2.1.2 다음 항목"),
        ("4.2.2.2 재료", "4.2.2.2.1 기술검토서에 기재된 재료인지 확인한다. 4.2.2.2.2 부속품 규격 확인."),
        ("4.2.2.3 접합", "4.2.2.3.1 용접방법을 확인한다. 4.2.2.3.2 외관검사 및 비파괴시험. 4.2.2.3.3 PE융착원 자격. 4.2.2.3.4 굴곡허용반경."),
        ("4.2.2.4 노출배관 및 교량에 설치된 배관", "배관의 손상 여부, 지지 및 신축흡수조치의 기능상 유해한 부식 등이 없는지 확인한다."),
        ("4.2.2.5 전기부식방지조치", "4.2.2.5.1 전기방식방법의 선택과 시공이 적정한지 확인한다. 4.2.2.5.2 관대지전위를 측정한다. 4.2.2.5.3 추가"),
        ("4.2.2.6 지하매설 배관 순회검사", "4.2.2.6.1 노면의 침하와 배관 방호조치 이상 유무를 확인한다. 4.2.2.6.2 라인마크 및 표지판을 확인한다."),
        ("4.2.2.7 가스차단장치", "4.2.2.7.1 설치위치를 확인한다. 4.2.2.7.2 손상 유무. 4.2.2.7.3 수동식 밸브 작동. 4.2.2.7.4 밸브박스 상태."),
        ("4.2.2.8 수취기", "4.2.2.8.1 손상. 4.2.2.8.2 중압 이상 밸브 작동·누출. 4.2.2.8.3 침수 여부."),
        ("4.2.2.9 기밀시험 또는 누출검사", "4.2.2.9.1 시공감리 때 누출과 시험가스 방출을 확인한다. 4.2.2.9.2 정기검사 때 시험시기가 도래하면 기밀시험을 하고 가스검지기 등을 사용한다. 4.2.2.9.3 상세 방법"),
        ("4.2.2.10 내압시험", "4.2.2.10.1 중압 이상의 배관은 최고사용압력의 1.5배로 시험한다. 4.2.2.10.2 압력강하 및 변형·파손 확인. 4.2.2.10.3 상세 방법"),
    ]
    chunks = []
    for chunk_id, (heading, content) in enumerate(clauses, start=1):
        if heading.startswith("4.1."):
            hierarchy = f"[FS551] 4 검사 기준 > 4.1 검사 항목 > {heading}"
        elif heading.startswith("4.2.2."):
            hierarchy = f"[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2 시공감리, 정기검사 및 수시검사 > {heading}"
        else:
            hierarchy = f"[FS551] 4 검사 기준 > {heading}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": 1,
                "doc_type": "CODE",
                "doc_code": "FS551",
                "title": "배관 기준",
                "hierarchy": hierarchy,
                "page": 80 + chunk_id,
                "filename": "FS551.pdf",
                "content": content,
                "score": 100.0 - chunk_id,
            }
        )
    chunks.append(
        {
            "chunk_id": 16,
            "document_id": 1,
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "배관 기준",
            "hierarchy": "[FS551] 82페이지",
            "page": 82,
            "filename": "FS551.pdf",
            "content": "KGS FS551 2024 (3) 1.8에 따른 배관 설치제한의 확인",
            "score": 84.0,
        }
    )
    scope_chunk = {
        "chunk_id": 17,
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 1 일반사항 > 1.1 적용범위",
        "page": 12,
        "filename": "FS551.pdf",
        "content": "이 기준은 일반도시가스사업의 가스공급시설 중 가스배관의 시설·기술·검사 및 진단에 적용한다.",
        "score": 99.0,
    }

    class InspectionDatabase:
        saved = None

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert codes == ["FS551"]
            if heading == "검사":
                assert limit == 200
                return chunks
            assert heading in {"4.1", "4.2", "4.2.2"}
            return []

        def search(self, query, limit, codes, domain):
            assert query == "1.8 배관 설치제한"
            assert limit == 100 and codes == ["FS551"] and domain == "CODE"
            return [item for item in chunks if "1.8에 따른 배관 설치제한" in item["content"]]

        def search_document_scopes(self, codes):
            assert codes == ["FS551"]
            return [scope_chunk]

        def save_exchange(self, *args):
            self.saved = args
            return 123

    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        InspectionDatabase(),
        ReasonerMustNotRun(),
    )
    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FS551 문서 기준으로 공사 전, 시공 중, 정기검사 단계의 "
                "배관 검사 절차를 단계별로 정리해줘"
            )
        )
    )

    references = {int(value) for value in re.findall(r"\[(\d+)\]", response.answer)}
    citation_numbers = {item.number for item in response.citations}
    assert response.intent == "procedure"
    assert len(response.citations) == 16
    assert all(item.excerpt.strip() for item in response.citations)
    assert references == citation_numbers
    assert "매년" not in response.answer
    assert "이상 발견 시 즉시 시정" not in response.answer
    assert "별도의 ‘공사 전’ 단계" in response.answer
    assert "1.8에 따른 배관 설치제한의 확인" in response.citations[1].excerpt
    assert response.citations[1].page == 82
    assert "4.2.2.1~4.2.2.3 및 4.2.2.10을 제외할 수 있습니다" in response.answer
    assert "기밀시험 시기가 도래한 경우" in response.answer
    assert "내압시험" in response.answer

    scope_and_procedure_response = await pipeline.run(
        ChatRequest(message="KGS FS551 적용범위와 주요 검사방법을 단계별로 정리해줘")
    )
    assert scope_and_procedure_response.intent == "procedure"
    assert scope_and_procedure_response.citations[0].page == 12
    assert "적용범위(1.1)" in scope_and_procedure_response.answer
    assert "주요 확인 항목" in scope_and_procedure_response.answer
    assert "설계도면의 종류·축척" not in scope_and_procedure_response.answer


def test_fs551_inspection_overview_recognizes_flat_database_hierarchies():
    required_codes = (
        "4.1.2", "4.1.3", "4.1.4", "4.2", "4.2.2",
        *(f"4.2.2.{number}" for number in range(1, 11)),
    )
    chunks = [
        {
            "chunk_id": number,
            "document_id": 1,
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "배관 기준",
            "hierarchy": f"[FS551] {code} 기준 항목",
            "page": 80 + number,
            "filename": "FS551.pdf",
            "content": f"{code} 기준 내용",
            "score": 1.0,
        }
        for number, code in enumerate(required_codes, start=1)
    ]

    result = RagPipeline._fs551_inspection_overview(chunks)

    assert result is not None
    answer, citations = result
    assert len(citations) == len(required_codes)
    assert "별도의 ‘공사 전’ 단계" in answer
    assert "정기검사" in answer and "시공감리" in answer


@pytest.mark.asyncio
async def test_fs551_contextual_periodic_tightness_followup_is_source_extracted():
    source = (
        "4.2.2.9.2 정기검사를 하는 때에는 기밀시험을 실시(기밀시험 시기가 도래한 경우에만 한다)하고, "
        "그 밖에 가스누출검지기를 이용하여 가스누출여부를 확인하여 이상이 있는 지하매설 배관에 대해서는 "
        "보링작업에 의한 누출검사를 실시한다. 4.2.2.9.3 다음 방법"
    )
    chunk = {
        "chunk_id": 1, "document_id": 1, "doc_type": "CODE", "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2 시공감리, 정기검사 및 수시검사 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94, "filename": "FS551.pdf", "content": source, "score": 1.0,
    }

    class FollowupDatabase:
        saved = None

        def history(self, _conversation_id):
            return [
                {"role": "user", "content": "FS551 검사 기준을 설명해줘"},
                {"role": "assistant", "content": "FS551 배관 기준을 확인했습니다."},
            ]

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            self.saved = args
            return 124

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FollowupDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message="그럼 정기검사 때 기밀시험은 매번 하는 거야?", conversation_id="c1"))

    assert "아니요" in response.answer
    assert "시기가 도래한 경우에만" in response.answer
    assert "보링작업" in response.answer
    assert response.intent == "fact"
    assert len(response.citations) == 1
    assert response.citations[0].page == 94
    assert re.findall(r"\[(\d+)\]", response.answer) == ["1"]


@pytest.mark.asyncio
async def test_fs551_branch_tee_report_exception_keeps_both_conditions():
    source = (
        "(3-1) 제품번호(Lot No 등)가 관리되고, 각 제품에 대하여 전문 비파괴검사업소에서 발행한 "
        "비파괴시험 성적서가 있는 경우에는 그 시험성적서를 비파괴시험에 갈음하여 현장에서 "
        "별도의 비파괴시험을 하지 않을 수 있다. (3-2) 제품번호(Lot No 등)가 관리되지 않아 "
        "비파괴 시험성적서로 확인이 곤란한 경우는 현장에서 비파괴시험을 실시한다. 4.2.2.3.3 PE배관"
    )
    chunk = {
        "chunk_id": 2, "document_id": 1, "doc_type": "CODE", "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2 시공감리, 정기검사 및 수시검사 > 4.2.2.3 접합",
        "page": 92, "filename": "FS551.pdf", "content": source, "score": 1.0,
    }

    class JoiningDatabase:
        saved = None

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "접합"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            self.saved = args
            return 125

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        JoiningDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="FS551 제품 분기티의 비파괴시험을 현장시험 대신 시험성적서로 갈음할 수 있는 조건은?"
        )
    )

    assert "제품번호(Lot No 등)가 관리되고" in response.answer
    assert "각 제품에 대하여 전문 비파괴검사업소" in response.answer
    assert "비파괴 시험성적서로 확인이 곤란한 경우" in response.answer
    assert "현장에서 비파괴시험을 실시한다" in response.answer
    assert "용접방법을 확인한다" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 92
    assert "제품번호(Lot No 등)가 관리되고" in response.citations[0].excerpt
    assert "제품번호(Lot No 등)가 관리되지 않아" in response.citations[0].excerpt


def test_fs551_low_pressure_branch_tee_keeps_80mm_all_weld_boundary_qualified():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20498,
        "hierarchy": "[FS551] 4.2.2.3 접합",
        "content": (
            "4.2.2.3.2 용접접합부는 외관검사 및 비파괴시험으로 결함유무를 확인하되, "
            "분기티 용접부에 대한 비파괴시험은 다음과 같이 실시한다. "
            "(1) 중압용 분기티의 경우에는 모든 용접부에 대하여 비파괴시험을 실시한다. "
            "(2) 저압용 분기티의 경우에는 분기되는 배관의 호칭지름이 80mm 이상인 분기티의 "
            "모든 용접부에 대하여 실시한다. (3) 제품으로 제작된 분기티"
        ),
    }
    query_65 = (
        "KGS FS551 저압용 분기티에서 분기되는 배관의 호칭지름이 65mm이면 "
        "모든 용접부에 비파괴시험을 해야 해?"
    )
    result_65 = RagPipeline._fs551_low_pressure_branch_tee_diameter(query_65, [chunk])

    assert result_65 is not None
    _source, answer_65, excerpt_65 = result_65
    assert "65mm는 80mm 미만" in answer_65
    assert "모든 용접부’ 시험 조건에는 해당하지 않습니다" in answer_65
    assert "비파괴시험 불필요’라고 단정할 수는 없습니다" in answer_65
    assert "외관검사 및 비파괴시험" in excerpt_65
    assert "호칭지름이 80mm 이상" in excerpt_65
    assert "<개정" not in excerpt_65

    natural_query_65 = (
        "KGS FS551 최고사용압력이 저압이고 호칭지름 65mm인 배관을 "
        "분기관에 T자 접합하면 용접부 비파괴검사를 전부 면제할 수 있어?"
    )
    natural_result_65 = RagPipeline._fs551_low_pressure_branch_tee_diameter(
        natural_query_65, [chunk]
    )
    assert natural_result_65 is not None
    assert "65mm는 80mm 미만" in natural_result_65[1]
    assert "비파괴시험 불필요’라고 단정할 수는 없습니다" in natural_result_65[1]

    query_80 = (
        "KGS FS551 저압용 분기티에서 분기되는 배관의 호칭지름이 80mm이면 "
        "모든 용접부에 비파괴시험을 해야 해?"
    )
    result_80 = RagPipeline._fs551_low_pressure_branch_tee_diameter(query_80, [chunk])
    assert result_80 is not None
    assert "예. 80mm는 80mm 이상" in result_80[1]
    assert "모든 용접부에 비파괴시험을 실시해야 합니다" in result_80[1]


@pytest.mark.asyncio
async def test_fs551_branch_tee_diameter_route_cites_boundary_without_llm():
    query = (
        "KGS FS551 저압용 분기티에서 분기되는 배관의 호칭지름이 65mm이면 "
        "모든 용접부에 비파괴시험을 해야 해?"
    )
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.3 접합",
        "page": 92,
        "filename": "FS551.pdf",
        "chunk_id": 20498,
        "content": (
            "4.2.2.3.2 용접접합부는 외관검사 및 비파괴시험으로 결함유무를 확인하되, "
            "분기티 용접부에 대한 비파괴시험은 다음과 같이 실시한다. "
            "(1) 중압용 분기티의 경우에는 모든 용접부에 대하여 비파괴시험을 실시한다. "
            "(2) 저압용 분기티의 경우에는 분기되는 배관의 호칭지름이 80mm 이상인 분기티의 "
            "모든 용접부에 대하여 실시한다. (3) 제품으로 제작된 분기티"
        ),
        "score": 1.0,
    }

    class JoiningDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "접합"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 2

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        JoiningDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "65mm는 80mm 미만" in response.answer
    assert "비파괴시험 불필요’라고 단정할 수는 없습니다" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 92
    assert "호칭지름이 80mm 이상" in response.citations[0].excerpt

    natural_query = (
        "KGS FS551 최고사용압력이 저압이고 호칭지름 65mm인 배관을 "
        "분기관에 T자 접합하면 용접부 비파괴검사를 전부 면제할 수 있어?"
    )
    natural_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        JoiningDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=natural_query))
    assert "65mm는 80mm 미만" in natural_response.answer
    assert "비파괴시험 불필요’라고 단정할 수는 없습니다" in natural_response.answer
    assert natural_response.citations[0].page == 92


@pytest.mark.asyncio
async def test_fs551_branch_tee_diameter_followup_inherits_prior_document_code():
    query = "그럼 호칭지름이 79.9mm인 저압 분기티는 어떻게 달라?"
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.3 접합",
        "page": 92,
        "filename": "FS551.pdf",
        "chunk_id": 20498,
        "content": (
            "4.2.2.3.2 용접접합부는 외관검사 및 비파괴시험으로 결함유무를 확인하되, "
            "분기티 용접부에 대한 비파괴시험은 다음과 같이 실시한다. "
            "(1) 중압용 분기티의 경우에는 모든 용접부에 대하여 비파괴시험을 실시한다. "
            "(2) 저압용 분기티의 경우에는 분기되는 배관의 호칭지름이 80mm 이상인 분기티의 "
            "모든 용접부에 대하여 실시한다."
        ),
        "score": 1.0,
    }

    class ContextJoiningDatabase:
        def history(self, _conversation_id):
            return [{
                "role": "user",
                "content": "KGS FS551 저압 분기티 호칭지름 80mm는 모든 용접부 비파괴시험 대상이야?",
            }]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "접합"
            assert codes == ["FS551"] and limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 3

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextJoiningDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query, conversation_id="branch-tee-followup"))

    assert "79.9mm는 80mm 미만" in response.answer
    assert "비파괴시험 불필요’라고 단정할 수는 없습니다" in response.answer
    assert [item.doc_code for item in response.citations] == ["FS551"]


@pytest.mark.asyncio
async def test_contextual_followup_without_prior_document_stops_before_cross_document_search():
    class NoContextDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def save_exchange(self, *args):
            return 4

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        NoContextDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message="그럼 호칭지름이 79.9mm인 저압 분기티는 어떻게 달라?")
    )

    assert response.intent == "clarification"
    assert "기준번호나 시설·배관 유형을 확인할 수 없어" in response.answer
    assert "무관한 기준을 섞어 답하지 않았습니다" in response.answer
    assert response.citations == []


@pytest.mark.asyncio
async def test_fs551_periodic_inspection_interval_does_not_borrow_other_test_cycle():
    source = (
        "배관에 대한 정기검사는 수요자에게 가스를 공급하기 위하여 분기되는 관경 50mm 이하인 저압의 "
        "공급관에 연결되는 사용자공급관을 제외한 배관에만 실시하며 항목은 다음과 같다. "
        "(1) 2.4.4에 따른 가스설비 설치와 점검의 확인"
    )
    chunk = {
        "chunk_id": 3, "document_id": 1, "doc_type": "CODE", "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4.1.3 정기검사",
        "page": 90, "filename": "FS551.pdf", "content": source, "score": 1.0,
    }
    table_of_contents_chunk = {
        **chunk,
        "chunk_id": 2,
        "page": 11,
        "hierarchy": "[FS551] 4.1.3 정기검사················78",
        "content": "4.1.3 정기검사 ........................................ 78",
    }

    class IntervalDatabase:
        saved = None

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "정기검사"
            assert codes == ["FS551"]
            assert limit == 100
            return [table_of_contents_chunk, chunk]

        def save_exchange(self, *args):
            self.saved = args
            return 126

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        IntervalDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message="FS551 배관 정기검사는 몇 년 주기인가?"))

    assert "정기검사 대상 배관과 검사 항목을 규정" in response.answer
    assert "간격은 이 조항에 명시되어 있지 않습니다" in response.answer
    assert not re.search(r"매년|연\s*\d+\s*회|\d+\s*년마다", response.answer)
    assert response.citations[0].hierarchy.endswith("4.1.3 정기검사")
    assert "사용자공급관을 제외한 배관에만 실시" in response.citations[0].excerpt
    assert "(1)" not in response.citations[0].excerpt


def test_comparison_answer_covers_and_labels_each_document():
    fs_quote = "이 기준은 가스배관의 시설·기술·검사 및 진단에 적용한다."
    fu_quote = "이 기준은 가스사용시설의 설치·운영 및 검사에 적용한다."
    citations = [
        Citation(
            number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
            title="배관 기준", hierarchy="[FS551] 1.1 적용범위", page=12,
            filename="FS551.pdf", excerpt=fs_quote, score=1.0,
        ),
        Citation(
            number=2, document_id=2, chunk_id=2, doc_type="CODE", doc_code="FU551",
            title="사용시설 기준", hierarchy="[FU551] 1.1 적용범위", page=13,
            filename="FU551.pdf", excerpt=fu_quote, score=1.0,
        ),
    ]
    grounded = GroundedClaims(
        claims=[
            EvidenceClaim(claim=fs_quote, citation_number=1, exact_quote=fs_quote),
            EvidenceClaim(claim=fu_quote, citation_number=2, exact_quote=fu_quote),
        ],
        insufficient_evidence=False,
    )

    answer, used = RagPipeline._validated_claim_answer(
        grounded,
        citations,
        [{"content": fs_quote}, {"content": fu_quote}],
        required_document_codes=("FS551", "FU551"),
        minimum_claims=2,
    )

    assert "FS551:" in answer
    assert "FU551:" in answer
    assert [item.doc_code for item in used] == ["FS551", "FU551"]


def test_scope_comparison_note_does_not_invent_a_physical_cutoff():
    citations = [
        Citation(
            number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FS551",
            title="배관 기준", hierarchy="[FS551] 1.1 적용범위", page=12,
            filename="FS551.pdf", excerpt="가스공급시설 중 가스배관", score=1.0,
        ),
        Citation(
            number=2, document_id=2, chunk_id=2, doc_type="CODE", doc_code="FU551",
            title="사용시설 기준", hierarchy="[FU551] 1.1 적용범위", page=13,
            filename="FU551.pdf", excerpt="가스사용시설의 설치·운영 및 검사", score=1.0,
        ),
    ]

    answer = _add_scope_boundary_note("적용범위 비교", citations)

    assert "물리적인 시설 경계" in answer
    assert "특정할 수 없습니다" in answer
    assert "[1] [2]" in answer


def test_scope_comparison_note_calls_out_unresolved_boundary_for_other_kgs_codes():
    citations = [
        Citation(
            number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="FP217",
            title="저장식 수소연료 충전 기준", hierarchy="[FP217] 1 일반사항 > 1.1 적용범위",
            page=12, filename="FP217.pdf", excerpt="배관 또는 저장설비로부터 공급받은 수소", score=1.0,
        ),
        Citation(
            number=2, document_id=2, chunk_id=2, doc_type="CODE", doc_code="FP216",
            title="제조식 수소연료 충전 기준", hierarchy="[FP216] 1 일반사항 > 1.1 적용범위",
            page=13, filename="FP216.pdf", excerpt="수소를 제조·압축하여 이동수단에 충전", score=1.0,
        ),
    ]

    answer = _add_scope_boundary_note("FP216과 FP217은 각각 제조식과 저장식이다. [1] [2]", citations)

    assert "복합·연계 시설" in answer
    assert "구체적인 적용 경계는 이 조항들만으로 확정할 수 없습니다" in answer
    assert answer.endswith("[1] [2]")


def test_scope_comparison_note_does_not_generalize_to_unverified_code_pairs():
    citations = [
        Citation(
            number=1, document_id=1, chunk_id=1, doc_type="CODE", doc_code="AA012",
            title="퓨즈콕 기준", hierarchy="[AA012] 1 일반사항 > 1.1 적용범위",
            page=9, filename="AA012.pdf", excerpt="퓨즈콕 기준 적용", score=1.0,
        ),
        Citation(
            number=2, document_id=2, chunk_id=2, doc_type="CODE", doc_code="FU671",
            title="수소연료사용시설 기준", hierarchy="[FU671] 1 일반사항 > 1.1 적용범위",
            page=11, filename="FU671.pdf", excerpt="수소연료사용시설 기준 적용", score=1.0,
        ),
    ]

    assert _add_scope_boundary_note("문서별 적용범위 비교", citations) == "문서별 적용범위 비교"


def test_fs551_fp216_boundary_answer_includes_equipment_carveout_but_no_physical_cutoff():
    query = (
        "KGS FS551과 FP216은 적용 대상이 어떻게 다르고, 한 시설에서 두 기준의 "
        "경계가 어디인지 문서만으로 특정할 수 있어?"
    )
    scope_chunks = [
        {
            "doc_code": "FS551",
            "hierarchy": "[FS551] 1 일반사항 > 1.1 적용범위",
            "content": "일반도시가스사업의 가스공급시설 중 가스배관에 적용한다.",
        },
        {
            "doc_code": "FP216",
            "hierarchy": "[FP216] 1 일반사항 > 1.1 적용범위",
            "content": "수소를 제조·압축하여 이동수단에 충전하는 제조식 수소연료 충전시설에 적용한다.",
        },
    ]
    definition_chunks = [
        {
            "doc_code": "FP216",
            "hierarchy": "[FP216] 1 일반사항 > 1.3 용어정의",
            "content": (
                "1.3.11.2 앞 조항의 설비와 연결된 비고압 수소설비. 다만, "
                "수소연료사용시설에 설치된 설비는 제외한다. 1.3.12 다음 용어"
            ),
        }
    ]

    result = RagPipeline._fs551_fp216_scope_boundary(
        query, ["FS551", "FP216"], scope_chunks, definition_chunks
    )

    assert result is not None
    source_rows, answer = result
    assert [row[0]["doc_code"] for row in source_rows] == ["FS551", "FP216", "FP216"]
    assert [row[2] for row in source_rows] == [12, 13, 13]
    assert "수소연료사용시설에 설치된 설비는 제외" in source_rows[2][1]
    assert "물리적 인계점을 특정 밸브·플랜지 등으로 정하지는 않습니다" in answer
    assert "이 조항들만으로 확정할 수 없습니다" in answer
    assert "[1] [2] [3]" in answer
    assert all(not re.search(r"<(?:개정|신설|삭\s*제)", row[1]) for row in source_rows)


def test_fp216_fp217_clearance_comparison_uses_both_clause_conditions():
    query = (
        "KGS FP216과 FP217에서 사업소경계까지의 기본 안전거리와 방호벽 설치 시 "
        "5m로 줄일 수 있는 설비 범위를 각각 비교해줘."
    )
    chunks = [
        {
            "doc_code": "FP217",
            "hierarchy": "[FP217] 2 시설기준 > 2.1 배치기준 > 2.1.4 사업소경계와의 거리",
            "content": (
                "저장설비, 처리설비, 압축가스설비 및 충전설비는 그 외면으로부터 사업소경계까지 "
                "10m 이상의 안전거리를 유지한다. 다만, 저장설비·처리설비 및 압축가스설비의 주위에 "
                "2.7.2.2에 따른 방호벽을 설치하는 경우에는 5m 이상의 안전거리를 유지할 수 있다."
            ),
        },
        {
            "doc_code": "FP216",
            "hierarchy": "[FP216] 2 시설기준 > 2.1 배치기준 > 2.1.4 사업소경계와의 거리",
            "content": (
                "저장설비, 처리설비, 압축가스설비 및 충전설비는 그 외면으로부터 사업소경계까지 "
                "10m 이상의 안전거리를 유지한다. 다만, 처리설비 및 압축가스설비의 주위에 "
                "2.7.2.2에 따른 방호벽을 설치하는 경우에는 5m 이상의 안전거리를 유지할 수 있다."
            ),
        },
    ]

    result = RagPipeline._fp216_fp217_boundary_clearance(
        query, ["FP216", "FP217"], chunks
    )

    assert result is not None
    sources, answer = result
    assert [chunk["doc_code"] for chunk in sources] == ["FP216", "FP217"]
    assert "기본 10m 이상" in answer
    assert "FP216은 2.7.2.2 방호벽을 처리설비·압축가스설비" in answer
    assert "FP217은 같은 조건의 예외 대상에 저장설비도 포함해 저장설비·처리설비·압축가스설비" in answer


def test_numbered_source_list_keeps_all_items_and_nested_clause_numbers():
    content = (
        "정기검사 항목은 다음과 같다. "
        "(1) 2.4.4에 따른 설비 확인 "
        "(2) 2.5.5.3에 따른 이음쇠 확인 "
        "(3) 2.5.8.3.1(1)에 따른 입상관 확인 "
        "(4) 3.1.8에 따른 유지관리 확인"
    )

    intro, items = RagPipeline._numbered_source_items(content)

    assert intro == "정기검사 항목은 다음과 같다."
    assert [number for number, _text in items] == [1, 2, 3, 4]
    assert "2.5.8.3.1(1)" in items[2][1]


def _fs551_periodic_test_chunk(item_count: int = 23) -> dict:
    source_items = [f"2.4.4에 따른 확인 항목 {number}" for number in range(1, item_count + 1)]
    if item_count >= 7:
        source_items[6] = "2.5.8.1.6에 따른 천정내부·바닥·벽속에 공급관 설치 여부 확인"
    if item_count >= 23:
        source_items[22] = "2.10.3에 따른 배관 색상 및 황색띠의 확인(노출배관에만 한다)"
    content = (
        "4.1.3 정기검사 배관에 대한 정기검사는 수요자에게 가스를 공급하기 위하여 "
        "분기되는 관경 50mm 이하인 저압의 공급관에 연결되는 사용자공급관을 제외한 "
        "배관에만 실시하며 항목은 다음과 같다. "
        + " ".join(
            f"({number}) {source_item}"
            for number, source_item in enumerate(source_items, start=1)
        )
    )
    return {
        "chunk_id": 26008,
        "document_id": 370,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "일반도시가스사업 배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.1 검사 항목 > 4.1.3 정기검사",
        "page": 90,
        "filename": "FS551.pdf",
        "content": content,
        "score": 100.0,
    }


@pytest.mark.asyncio
async def test_fs551_periodic_inspection_list_uses_full_ordered_source_and_never_calls_llm():
    source_chunk = _fs551_periodic_test_chunk()
    table_of_contents_chunk = {
        **source_chunk,
        "chunk_id": 25664,
        "page": 11,
        "hierarchy": "[FS551] 4.1.3 정기검사················78",
        "content": "4.1.3 정기검사 ........................................ 78",
    }

    class PeriodicDatabase:
        saved = None

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "4.1.3"
            assert codes == ["FS551"] and limit == 100
            return [table_of_contents_chunk, source_chunk]

        def save_exchange(self, *args):
            self.saved = args
            return 456

    database = PeriodicDatabase()
    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        database,
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message=(
                "KGS FS551 배관 정기검사 항목을 빠짐없이 항목별로 정리하고, "
                "검사 대상에서 제외되는 배관 조건도 구분해줘."
            )
        )
    )

    assert response.intent == "fact"
    assert len(response.citations) == 1
    assert "관경 50mm 이하인 저압의 공급관에 연결되는 사용자공급관" in response.answer
    listed_numbers = [
        int(value)
        for value in re.findall(r"^\((\d+)\)", response.answer, flags=re.MULTILINE)
    ]
    assert listed_numbers == list(range(1, 24))
    assert "2.5.8.1.6에 따른 천정내부·바닥·벽속에 공급관 설치 여부 확인" in response.answer
    assert "2.10.3에 따른 배관 색상 및 황색띠" in response.answer
    assert "(23)" in response.citations[0].excerpt
    assert database.saved[4] == "source-list-FS551-periodic-inspection"


@pytest.mark.asyncio
async def test_fs551_periodic_inspection_list_fails_closed_when_index_is_truncated():
    source_chunk = _fs551_periodic_test_chunk(item_count=11)

    class PeriodicDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            return [source_chunk]

        def save_exchange(self, *args):
            return 457

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        PeriodicDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message="KGS FS551 정기검사 전체 항목을 빠짐없이 나열해줘.")
    )

    assert "전체 연속 목록을 색인에서 확인하지 못해" in response.answer
    assert "전체 목록처럼 제시하지 않았습니다" in response.answer
    assert response.citations == []


@pytest.mark.asyncio
async def test_fs551_periodic_inspection_scope_followup_distinguishes_branch_and_user_pipe():
    source_chunk = _fs551_periodic_test_chunk()

    class PeriodicDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "4.1.3"
            assert codes == ["FS551"] and limit == 100
            return [source_chunk]

        def save_exchange(self, *args):
            return 458

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        PeriodicDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 50 mm 이하인 저압 공급관 자체도 정기검사에서 제외된다는 뜻이야?",
            history=[
                ChatTurn(role="user", content="KGS FS551 정기검사 항목은 무엇인가요?"),
                ChatTurn(role="assistant", content="정기검사 항목과 대상 제외 조건을 안내했습니다."),
            ],
        )
    )

    assert response.intent == "fact"
    assert "아니요." in response.answer
    assert "공급관 자체가 아니라" in response.answer
    assert "연결되는 사용자공급관" in response.answer
    assert "50 mm는 50 mm 이하 조건에 포함됩니다" in response.answer
    assert len(response.citations) == 1
    assert "사용자공급관을 제외한" in response.citations[0].excerpt
    assert "(1)" not in response.citations[0].excerpt

    numeric_result = RagPipeline._fs551_periodic_inspection_scope_boundary(
        "KGS FS551에서 관경이 정확히 50 mm와 50.1 mm인 경우, "
        "정기검사 제외 조건 중 수치 기준은 어떻게 달라져?",
        [source_chunk],
    )
    assert numeric_result is not None
    assert "50 mm는 50 mm 이하 조건에 포함됩니다" in numeric_result[1]
    assert "50.1 mm는 50 mm 이하 조건을 충족하지 않습니다" in numeric_result[1]
    assert not numeric_result[1].startswith("아니요.")

    decimal_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        PeriodicDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message=(
                "KGS FS551 정기검사에서 관경 50.0mm와 50.1mm의 제외 조건을 "
                "비교해줘."
            )
        )
    )
    assert decimal_response.intent == "fact"
    assert "50 mm 이하 조건에 포함됩니다" in decimal_response.answer
    assert "50 mm 이하 조건을 충족하지 않습니다" in decimal_response.answer


def test_pe_pipe_leak_test_schedule_is_read_directly_from_source_table():
    query = "그 기준에서 PE배관 정기 기밀시험은 언제부터 몇 년마다 실시하나요?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20505,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": "대상구분 기밀시험 실시시기 PE배관 설치 후 15년이 되는 해및그 이후 5년마다",
    }

    result = RagPipeline._pe_leak_test_schedule(query, [chunk])

    assert result is not None
    source_chunk, schedule, excerpt = result
    assert source_chunk["chunk_id"] == 20505
    assert schedule == "PE배관의 정기 기밀시험 시기는 PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다입니다."
    assert excerpt == "PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다"


def test_pe_pipe_schedule_applies_installation_year_arithmetically():
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20505,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": "PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다",
    }

    result = RagPipeline._pe_leak_test_schedule(
        "KGS FS551에서 2012년에 설치한 PE배관의 정기 기밀시험은 언제 처음 하나요?",
        [chunk],
    )

    assert result is not None
    _source_chunk, answer, _excerpt = result
    assert "최초 실시 연도는 2027년" in answer
    assert "2032년부터 5년마다" in answer

    followup = RagPipeline._pe_leak_test_schedule(
        "그럼 그 다음 검사는 몇 년도야?",
        [chunk],
        "user: KGS FS551에서 2012년에 설치한 PE배관 정기 기밀시험은 언제 처음 해? "
        "assistant: 최초 실시 연도는 2027년이고 이후 2032년부터 5년마다입니다.",
    )
    assert followup is not None
    assert "그 다음 시험은 2032년" in followup[1]
    assert RagPipeline._pe_leak_test_schedule(
        "그럼 가스 검지기 위치는 어디야?", [chunk], "PE배관 기밀시험 주기 관련 대화"
    ) is None


def test_pe_pipe_schedule_extractor_requires_matching_question_and_clause():
    chunk = {
        "doc_type": "CODE",
        "chunk_id": 1,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": "PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다",
    }

    assert RagPipeline._pe_leak_test_schedule("PE배관이 무엇인가요?", [chunk]) is None
    assert RagPipeline._pe_leak_test_schedule(
        "PE배관 기밀시험 주기는?",
        [{**chunk, "hierarchy": "[FS551] 2.5.2 재료"}],
    ) is None


def test_fs551_tightness_interval_table_keeps_every_pipe_category_and_condition():
    common = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
    }
    top_chunk = {
        **common,
        "chunk_id": 20505,
        "content": (
            "표 4.2.2.9.5(2) 대상구분 기밀시험 실시시기 "
            "PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다 "
            "폴리에틸렌 피복강관 1993년 6월 26일 이후에 설치된 것 "
            "1993년 6월 25일 이전에 설치된 것 설치 후 15년이 되는 해 및 그 이후 "
            "3년마다(다만, 정밀안전진단을 받은 경우 그 이후 3년으로 한다) 300m3 미만"
        ),
    }
    bottom_chunk = {
        **common,
        "chunk_id": 20506,
        "content": (
            "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다 "
            "공동주택 등(다세대주택 제 외)의 부지내에 설치된 배 관 "
            "(3-3-2-2)에 따라 검지 공을 설치하고 도시가스사 업자가 매년 자체점검을 실 시한 배관 6년마다 "
            "그 밖의 배관 설치 후 15년이 되는 해까지 5년마다, 15년 경과 "
            "31년이 되는 해까지 4년마다, 31년 경과 3년마다 [비고] 기밀시험 실시시기는 마지막 기밀시험일을 기준으로 산정한다."
        ),
    }

    result = RagPipeline._fs551_tightness_interval_table(
        "그럼 기밀시험 주기는 어떻게 돼?",
        [top_chunk, bottom_chunk],
        "user: KGS FS551 배관 정기검사는 몇 년 주기인가?",
    )

    assert result is not None
    sources, answer = result
    assert [(row[0]["chunk_id"], row[2]) for row in sources] == [(20505, 96), (20506, 97)]
    assert "PE배관: 설치 후 15년이 되는 해부터, 이후 5년마다" in answer
    assert "피복강관" in answer and "1993년 6월 26일 이후" in answer
    assert "정밀안전진단" in answer and "3년마다" in answer
    assert "그 밖의 배관: 설치 후 15년이 되는 해부터 1년마다" in answer
    assert "매년 자체점검한 배관: 6년마다" in answer
    assert "다세대주택 제외" in answer
    assert "15년 경과 후 31년이 되는 해까지 4년마다" in answer
    assert "마지막 기밀시험일을 기준" in answer
    assert "8.4kPa" not in answer and "24시간" not in answer
    assert "다세대주택 제외" in sources[1][1]
    assert "검지공을 설치하고 도시가스사업자가 매년 자체점검을 실시한 배관" in sources[1][1]
    assert "마지막 기밀시험일을 기준으로 산정한다" in sources[1][1]
    assert "검지 공" not in sources[1][1] and "제 외" not in sources[1][1]

    coated = RagPipeline._fs551_coated_steel_tightness_schedule(
        "KGS FS551 폴리에틸렌 피복강관 기밀시험 주기는?", [top_chunk]
    )
    assert coated is not None
    _source, coated_answer, coated_excerpt = coated
    assert "두 행에 공통으로 표시된 주기는" in coated_answer
    assert "15년이 되는 해 및 그 이후 3년마다" in coated_answer
    assert "정밀안전진단" in coated_answer
    assert "PE배관 설치 후" not in coated_excerpt
    assert "1993년 6월 26일 이후" in coated_excerpt
    assert "1993년 6월 25일 이전" in coated_excerpt

    assert RagPipeline._fs551_tightness_interval_table(
        "그럼 기밀시험 주기는 어떻게 돼?", [top_chunk, bottom_chunk], "이전 질문의 기준 정보 없음"
    ) is None
    assert RagPipeline._fs551_tightness_interval_table(
        "FS551 PE배관 기밀시험 주기는?", [top_chunk, bottom_chunk]
    ) is None
    assert RagPipeline._fs551_tightness_interval_table(
        "FS551 기밀시험 압력은?", [top_chunk, bottom_chunk]
    ) is None


def test_fs551_other_pipe_interval_returns_only_requested_row():
    chunk = {
        "chunk_id": 20506,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "hierarchy": "[FS551] 97페이지",
        "content": (
            "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다 "
            "공동주택 등(다세대주택 제외)의 부지내에 설치된 배관 "
            "6년마다 [비고] 기밀시험 실시시기는 마지막 기밀시험일을 기준으로 산정한다."
        ),
    }
    result = RagPipeline._fs551_other_pipe_tightness_schedule(
        "KGS FS551에서 그 밖의 배관 기밀시험 주기는?", [chunk]
    )

    assert result is not None
    source, answer, excerpt = result
    assert source["chunk_id"] == 20506
    assert "설치 후 15년이 되는 해 및 그 이후 1년마다" in answer
    assert excerpt == "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다"


def test_fs551_user_supply_and_other_interval_explains_scope_exclusion_and_row():
    scope = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20507,
        "hierarchy": "[FS551] 4.1.3 정기검사",
        "page": 90,
        "content": (
            "4.1.3 정기검사 배관에 대한 정기검사는 수요자에게 가스를 공급하기 위하여 "
            "분기되는 관경 50mm 이하인 저압의 공급관에 연결되는 사용자공급관을 제외한 배관에만 실시하며"
        ),
    }
    other = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20508,
        "hierarchy": "[FS551] 97페이지",
        "page": 97,
        "content": "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다 공동주택 등 [비고] 기밀시험 실시시기는 마지막 기밀시험일을 기준으로 산정한다.",
    }

    result = RagPipeline._fs551_user_supply_and_other_tightness_schedule(
        "KGS FS551 사용자공급관과 그 밖의 배관 기밀시험 주기는?",
        [other],
        [scope],
    )

    assert result is not None
    sources, answer = result
    assert [source[0]["chunk_id"] for source in sources] == [20507, 20508]
    assert "사용자공급관" in answer and "제외" in answer
    assert "그 밖의 배관" in answer and "1년마다" in answer
    assert sources[0][1] == 90 and sources[1][1] == 97


@pytest.mark.asyncio
async def test_fs551_contextual_tightness_interval_followup_uses_complete_source_table():
    common = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
    }
    chunks = [
        {
            **common,
            "chunk_id": 20505,
            "content": (
                "PE배관 설치 후 15년이 되는 해 및 그 이후 5년마다 "
                "폴리에틸렌 피복강관 1993년 6월 26일 이후에 설치된 것 "
                "1993년 6월 25일 이전에 설치된 것 설치 후 15년이 되는 해 및 그 이후 "
                "3년마다(다만, 정밀안전진단을 받은 경우 그 이후 3년으로 한다) 300m3 미만"
            ),
        },
        {
            **common,
            "chunk_id": 20506,
            "content": (
                "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다 "
                "공동주택 등(다세대주택 제 외)의 부지내에 설치된 배 관 "
                "(3-3-2-2)에 따라 검지 공을 설치하고 도시가스사 업자가 매년 자체점검을 실 시한 배관 6년마다 "
                "그 밖의 배관 설치 후 15년이 되는 해까지 5년마다, 15년 경과 "
                "31년이 되는 해까지 4년마다, 31년 경과 3년마다 [비고] 기밀시험 실시시기는 마지막 기밀시험일을 기준으로 산정한다."
            ),
        },
    ]

    class IntervalDatabase:
        saved = None

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FS551"]
            assert limit == 100
            return chunks

        def save_exchange(self, *args):
            self.saved = args
            return 127

    database = IntervalDatabase()
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        database,
        ReasonerMustNotRun(),
    )
    response = await pipeline.run(
        ChatRequest(
            message="그럼 기밀시험 주기는 어떻게 돼?",
            history=[ChatTurn(role="user", content="KGS FS551 배관 정기검사는 몇 년 주기인가?")],
        )
    )

    assert "PE배관" in response.answer and "1년마다" in response.answer
    assert "6년마다" in response.answer and "31년이 되는 해까지 4년마다" in response.answer
    assert [citation.page for citation in response.citations] == [96, 97]
    assert [citation.chunk_id for citation in response.citations] == [20505, 20506]
    assert all(citation.excerpt for citation in response.citations)
    assert "마지막 기밀시험일을 기준으로 산정한다" in response.citations[1].excerpt
    assert re.findall(r"\[(\d+)\]", response.answer) == ["1", "1", "2", "2", "2"]

    # A short follow-up may omit both “기밀시험” and “주기”; retain the
    # preceding FS551 interval context and answer only the requested row.
    class ContextualOtherPipeDatabase:
        def history(self, _conversation_id):
            return [
                {"role": "user", "content": "KGS FS551 배관 종류별 정기 기밀시험 주기를 표로 정리해줘."},
                {"role": "assistant", "content": "FS551 표 4.2.2.9.5(2)에 따라 종류별 주기가 다릅니다."},
            ]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert codes == ["FS551"]
            assert limit == 100
            if heading == "정기검사":
                return [
                    {
                        "doc_type": "CODE",
                        "doc_code": "FS551",
                        "document_id": 1,
                        "title": "배관 기준",
                        "chunk_id": 20507,
                        "hierarchy": "[FS551] 4.1.3 정기검사",
                        "page": 90,
                        "filename": "FS551.pdf",
                        "score": 1.0,
                        "content": "4.1.3 정기검사 배관에 대한 정기검사는 사용자공급관을 제외한 배관에만 실시한다.",
                    }
                ]
            return [
                {
                    "doc_type": "CODE",
                    "doc_code": "FS551",
                    "document_id": 1,
                    "title": "배관 기준",
                    "chunk_id": 20506,
                    "hierarchy": "[FS551] 97페이지",
                    "page": 97,
                    "filename": "FS551.pdf",
                    "score": 1.0,
                    "content": "그 밖의 배관 설치 후 15년이 되는 해 및 그 이후 1년마다 [비고] 기밀시험 실시시기는 마지막 기밀시험일을 기준으로 산정한다.",
                }
            ]

        def search_pages(self, pages, codes, limit=8):
            return []

        def save_exchange(self, *args):
            return 128

    contextual_other_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualOtherPipeDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 그 밖의 배관은?",
            conversation_id="fs551-contextual-other-pipe",
        )
    )
    assert "설치 후 15년이 되는 해 및 그 이후 1년마다" in contextual_other_response.answer
    assert contextual_other_response.citations[0].page == 97

    universal_annual_response = await pipeline.run(
        ChatRequest(message="KGS FS551 모든 배관의 정기 기밀시험을 매년 해야 해?")
    )
    assert "배관 종류·조건별로 다릅니다" in universal_annual_response.answer
    assert "PE배관: 설치 후 15년이 되는 해부터, 이후 5년마다" in universal_annual_response.answer
    assert "그 밖의 배관: 설치 후 15년이 되는 해부터 1년마다" in universal_annual_response.answer
    assert "모든 배관의 공통 주기" not in universal_annual_response.answer
    assert [citation.page for citation in universal_annual_response.citations] == [96, 97]
    assert "마지막 기밀시험일을 기준으로 산정한다" in universal_annual_response.citations[1].excerpt

    coated_response = await pipeline.run(
        ChatRequest(message="KGS FS551에서 폴리에틸렌 피복강관은 설치연도에 따라 기밀시험 시기가 어떻게 달라져?")
    )
    assert "두 행에 공통으로 표시된 주기는" in coated_response.answer
    assert "그 이후 3년마다" in coated_response.answer
    assert "8.4kPa" not in coated_response.answer
    assert len(coated_response.citations) == 1
    assert coated_response.citations[0].page == 96
    assert "1993년 6월 26일 이후" in coated_response.citations[0].excerpt

    coated_after_pe_context_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FS551 폴리에틸렌 피복강관이 1990년 설치됐고 실제 마지막 기밀시험을 "
                "2006년에 했다면 다음 시험은 몇 년도야?"
            ),
            history=[
                ChatTurn(role="user", content="KGS FS551 PE배관 기밀시험 주기는 몇 년이야?"),
                ChatTurn(role="assistant", content="PE배관은 설치 후 15년부터 5년마다입니다."),
            ],
        )
    )
    assert "최초 주기도래는 2005년" in coated_after_pe_context_response.answer
    assert "실제 마지막 기밀시험이 2006년에 실시되었으므로" in coated_after_pe_context_response.answer
    assert "다음 예정연도는 2009년" in coated_after_pe_context_response.answer
    assert "2010년" not in coated_after_pe_context_response.answer
    assert "실제 후속 시험일은 마지막 기밀시험일을 기준으로 산정합니다" not in coated_after_pe_context_response.answer
    assert [citation.page for citation in coated_after_pe_context_response.citations] == [96, 97]

    coated_calculation_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FS551 폴리에틸렌 피복강관은 기밀시험 표에서 몇 년 주기야? "
                "검사 시점은 어떻게 계산해?"
            )
        )
    )
    assert "그 이후 3년마다" in coated_calculation_response.answer
    assert "마지막 기밀시험일을 기준으로 산정합니다" in coated_calculation_response.answer
    assert [citation.page for citation in coated_calculation_response.citations] == [96, 97]

    prior_coated_conversation = [
        {
            "role": "user",
            "content": (
                "KGS FS551에서 1990년에 설치한 폴리에틸렌 피복강관의 "
                "최초 기밀시험 시기와 이후 주기를 알려줘."
            ),
        },
        {
            "role": "assistant",
            "content": "1990년 설치분의 최초 주기 도래연도는 2005년이며 이후 3년마다입니다.",
        },
        {"role": "user", "content": "그럼 그 다음 기밀시험은 몇 년도야?"},
        {
            "role": "assistant",
            "content": (
                "FS551 표의 정기 기밀시험 시기는 배관 종류·조건별로 다릅니다. "
                "PE배관은 5년마다이며 폴리에틸렌 피복강관은 설치 후 15년 및 이후 3년마다입니다."
            ),
        },
    ]

    class CoatedRegressionDatabase(IntervalDatabase):
        def history(self, _conversation_id):
            return prior_coated_conversation

    coated_regression_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        CoatedRegressionDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 그 다음 기밀시험은 몇 년도야?",
            conversation_id="coated-followup-after-table",
        )
    )
    assert "첫 예정주기는 2005년" in coated_regression_response.answer
    assert "다음 예정연도는 2008년" in coated_regression_response.answer
    assert "2010년" not in coated_regression_response.answer
    assert [citation.page for citation in coated_regression_response.citations] == [96, 97]

    next_coated_exam_response = await pipeline.run(
        ChatRequest(
            message="그럼 그 다음 기밀시험은 몇 년도야?",
            history=[
                ChatTurn(
                    role="user",
                    content=(
                        "KGS FS551에서 1990년에 설치한 폴리에틸렌 피복강관의 "
                        "최초 기밀시험 시기와 이후 주기를 알려줘."
                    ),
                ),
                ChatTurn(
                    role="assistant",
                    content=(
                        "1990년 설치분의 최초 주기 도래연도는 2005년이며 이후 3년마다입니다."
                    ),
                ),
            ],
        )
    )
    assert "첫 예정주기는 2005년" in next_coated_exam_response.answer
    assert "다음 예정연도는 2008년" in next_coated_exam_response.answer
    assert "마지막 기밀시험일을 기준으로 산정" in next_coated_exam_response.answer
    assert [citation.page for citation in next_coated_exam_response.citations] == [96, 97]
    assert [citation.chunk_id for citation in next_coated_exam_response.citations] == [20505, 20506]

    yearless_coated_followup = await pipeline.run(
        ChatRequest(
            message="그 다음 시험은 몇 년 뒤야?",
            history=[
                ChatTurn(
                    role="user",
                    content="KGS FS551 폴리에틸렌 피복강관의 기밀시험 주기는 어떻게 돼?",
                ),
                ChatTurn(
                    role="assistant",
                    content="설치 후 15년이 되는 해 및 그 이후 3년마다입니다.",
                ),
            ],
        )
    )
    assert "첫 정기 기밀시험 이후 다음 시험까지의 간격은 3년입니다" in yearless_coated_followup.answer
    assert "마지막 기밀시험일을 기준으로 산정" in yearless_coated_followup.answer
    assert [citation.page for citation in yearless_coated_followup.citations] == [96, 97]

    next_exam_response = await pipeline.run(
        ChatRequest(
            message="그 다음 시험은 몇 년이야?",
            history=[
                ChatTurn(
                    role="user",
                    content="KGS FS551에서 2012년에 설치한 PE배관의 정기 기밀시험은 언제 처음 하나요?",
                )
            ],
        )
    )
    assert "최초 실시 연도는 2027년" in next_exam_response.answer
    assert "그 다음 시험은 2032년" in next_exam_response.answer
    assert "이후 5년마다" in next_exam_response.answer
    assert next_exam_response.citations[0].page == 96

    universal_response = await pipeline.run(
        ChatRequest(
            message="그럼 1년마다 주기는 모든 배관에 적용돼?",
            history=[
                ChatTurn(role="user", content="KGS FS551 기밀시험 주기를 배관 종류별로 알려줘."),
                ChatTurn(
                    role="assistant",
                    content=(
                        "FS551 표 4.2.2.9.5(2)의 정기 기밀시험 시기는 배관 종류·조건별로 다릅니다. "
                        "PE배관은 5년마다, 그 밖의 배관은 설치 후 15년이 되는 해부터 1년마다입니다."
                    ),
                ),
            ],
        )
    )
    assert "아니요" in universal_response.answer
    assert "‘그 밖의 배관’으로 구분한 행" in universal_response.answer
    assert "모든 배관의 공통 주기는 아닙니다" in universal_response.answer
    assert "PE배관 5년" in universal_response.answer
    assert [citation.page for citation in universal_response.citations] == [96, 97]

    multi_category_response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FS551 배관 종류별 정기 기밀시험 주기를 표로 정리해줘. "
                "PE배관, 폴리에틸렌 피복강관, 그 밖의 배관을 구분해줘."
            )
        )
    )
    assert "PE배관: 설치 후 15년이 되는 해부터, 이후 5년마다" in multi_category_response.answer
    assert "폴리에틸렌 피복강관" in multi_category_response.answer
    assert "그 밖의 배관: 설치 후 15년이 되는 해부터 1년마다" in multi_category_response.answer
    assert [citation.page for citation in multi_category_response.citations] == [96, 97]

    contextual_other_response = await pipeline.run(
        ChatRequest(
            message="그 밖의 배관은 설치 후 언제부터 몇 년마다야?",
            history=[
                ChatTurn(
                    role="user",
                    content="KGS FS551 배관 종류별 정기 기밀시험 주기를 표로 정리해줘.",
                ),
                ChatTurn(
                    role="assistant",
                    content="FS551 표의 배관 종류별 정기 기밀시험 주기를 정리했습니다.",
                ),
            ],
        )
    )
    assert "그 밖의 배관" in contextual_other_response.answer
    assert "설치 후 15년이 되는 해 및 그 이후 1년마다" in contextual_other_response.answer
    assert contextual_other_response.citations[0].page == 97


@pytest.mark.asyncio
async def test_fs551_early_tightness_test_requires_kgs_consultation_and_cites_table_page():
    source = (
        "4.2.2.9.5(2) 기밀시험 실시 시기는 표와 같다. 다만, 기밀시험 실시 시기 이전에 "
        "기밀시험을 하려는 경우에는 한국가스안전공사가 검사 신청인과 협의하여 "
        "기밀시험 실시 시기를 따로 정할 수 있다."
    )
    chunk = {
        "chunk_id": 20505,
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "content": source,
        "score": 1.0,
    }

    class EarlyTestDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 128

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        EarlyTestDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(message="KGS FS551에서 표에 정해진 기밀시험 시기보다 일찍 시험하려면 어떻게 해야 해?")
    )

    assert "검사 신청인이 임의로 앞당기는 것이 아니라" in response.answer
    assert "한국가스안전공사가 검사 신청인과 협의하여" in response.answer
    assert response.citations[0].page == 96
    assert "검사 신청인과 협의하여" in response.citations[0].excerpt


def test_tightness_test_pressure_extractor_preserves_30_kpa_exception_scope():
    query = "KGS FS551 기밀시험 pressure는 얼마고 30 kPa 이하 예외는?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": (
            "4.2.2.9.3(1-1) ... 4.2.2.9.4(1)이나 4.2.2.9.4(2)에 따른 방법. "
            "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력이상으로 실시한다. "
            "다만, 다음 기준에 해당하는 경우에는 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력이상으로 실시하지 않을 수 있다. "
            "(2-1) 최고사용압력이 저압인 배관 및그 부대설비 이외의 것으로서 최고사용압력이 30kPa 이하인 것은 "
            "시험압력을 최고사용압력으로 할수 있다. "
            "(2-2) 이미 설치된 사용자공급관은 시험압력을 사용압력 이상으로 할수 있다. "
            "(3) 기밀시험은 취성 파괴를 일으킬 우려가 없는 온도에서 실시한다. "
            "4.2.2.9.6 기밀시험을 생략할 수 있는 가스공급시설은 최고사용압력이 0MPa 이하의 것 "
            "또는 항상 대기로 개방되어 있는 것으로 한다."
        ),
    }

    result = RagPipeline._tightness_test_pressure_rule(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 20504
    assert "1.1배 또는 8.4kPa 중 높은" in answer
    assert "저압인 배관 및 그 부대설비 이외의 것으로서" in answer
    assert "30kPa 이하" in answer and "시험압력을 최고사용압력으로 할 수 있다" in answer
    assert "4.2.2.9.3(2-1)" in answer
    assert "이미 설치된 사용자공급관" not in answer
    assert "30kPa 이하" in excerpt

    numeric_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력 0.5MPa인 배관의 기밀시험압력을 계산해줘.",
        [chunk],
    )
    assert numeric_result is not None
    assert "0.5MPa의 1.1배는 0.55MPa" in numeric_result[1]
    assert "기밀시험압력은 550kPa (0.55MPa) 이상" in numeric_result[1]
    assert "다만, 다음 기준에 해당하는 경우" not in numeric_result[1]
    low_numeric_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력이 5kPa인 저압 배관의 기밀시험압력을 계산해줘. "
        "1.1배 계산값과 8.4kPa 중 어느 값이 기준인지 보여줘.",
        [chunk],
    )
    assert low_numeric_result is not None
    assert "1.1배는 0.0055MPa (5.5kPa)입니다" in low_numeric_result[1]
    assert "계산값 5.5kPa가 8.4kPa보다 낮으므로" in low_numeric_result[1]
    assert "기밀시험압력은 8.4kPa (0.0084MPa) 이상" in low_numeric_result[1]

    multi_numeric_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력 0.08MPa와 0.1MPa일 때 각각 기밀시험압력을 계산해줘.",
        [chunk],
    )
    assert multi_numeric_result is not None
    assert "최고사용압력 0.08MPa의 1.1배" in multi_numeric_result[1]
    assert "추가 입력값: 최고사용압력 0.1MPa의 1.1배" in multi_numeric_result[1]
    assert "기밀시험압력은 110kPa (0.11MPa) 이상" in multi_numeric_result[1]

    boundary_multi_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력 29.9kPa, 30kPa, 30.1kPa일 때 기밀시험압력과 "
        "30kPa 이하 예외 적용을 각각 계산해줘.",
        [chunk],
    )
    assert boundary_multi_result is not None
    assert boundary_multi_result[1].count("최고사용압력 0.03MPa의 1.1배") == 1
    assert "최고사용압력 0.0299MPa의 1.1배" in boundary_multi_result[1]
    assert "최고사용압력 0.0301MPa의 1.1배" in boundary_multi_result[1]
    assert "기밀시험압력은 32.89kPa (0.03289MPa) 이상" in boundary_multi_result[1]
    assert "기밀시험압력은 33.11kPa (0.03311MPa) 이상" in boundary_multi_result[1]
    assert "입력값이 30 kPa를 초과하므로" in boundary_multi_result[1]

    single_threshold_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력 29.9kPa일 때 기밀시험압력과 30kPa 예외를 알려줘.",
        [chunk],
    )
    assert single_threshold_result is not None
    assert "최고사용압력 0.0299MPa의 1.1배" in single_threshold_result[1]
    assert "최고사용압력 0.03MPa의 1.1배" not in single_threshold_result[1]
    assert "추가 입력값" not in single_threshold_result[1]

    scope_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 기밀시험은 30 kPa 이하 예외가 있으니 저압 배관도 모두 적용해도 돼?",
        [chunk],
    )
    assert scope_result is not None
    assert scope_result[1].startswith("아니요.")
    assert "저압인 배관과 그 부대설비를 제외합니다" in scope_result[1]
    assert "기본 시험압력은" in scope_result[1]
    assert "최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상입니다" in scope_result[1]
    assert "다만" not in scope_result[1]
    assert scope_result[1].count("30 kPa 예외") == 1
    natural_scope_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력이 저압인 배관에도 4.2.2.9.3(2-1)의 30 kPa 시험압력 예외를 적용해 시험압력을 낮출 수 있어?",
        [chunk],
    )
    assert natural_scope_result is not None
    assert natural_scope_result[1].startswith("아니요.")
    assert "저압 배관에는 이 예외를 근거로 시험압력을 최고사용압력까지 낮출 수 없습니다" in natural_scope_result[1]
    assert "기본 시험압력은" in natural_scope_result[1]
    omission_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 저압 배관에서 압력이 30 kPa 이하인 경우 기밀시험을 생략할 수 있어?",
        [chunk],
    )
    assert omission_result is not None
    assert "기밀시험 자체를 생략하는 규정이 아니라" in omission_result[1]
    assert "이 30 kPa 조건만으로 저압 배관의 기밀시험을 생략할 수 없습니다" in omission_result[1]
    assert "이상 이상" not in omission_result[1]
    numeric_low_pressure_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 최고사용압력이 25kPa인 저압 배관의 기밀시험압력을 계산해줘. "
        "30kPa 이하라서 시험압력을 25kPa로 낮춰도 돼?",
        [chunk],
    )
    assert numeric_low_pressure_result is not None
    assert "저압 배관에는 이 예외를 근거로 시험압력을 최고사용압력까지 낮출 수 없습니다" in numeric_low_pressure_result[1]
    assert "25kPa의 1.1배는 27.5kPa" in numeric_low_pressure_result[1]
    assert "기밀시험압력은 27.5kPa 이상" in numeric_low_pressure_result[1]
    assert RagPipeline._tightness_test_pressure_rule(
        "KGS FS551과 KGS FU551의 기밀시험 압력 기준 차이를 비교해줘.", [chunk]
    ) is None

    split_anchor = dict(chunk)
    split_anchor.update(
        chunk_id=20505,
        page=94,
        hierarchy="[FS551] 4.2.2.9.3 배관의 기밀시험 방법은 다음과 같다.",
        content=(
            "4.2.2.9.3 (2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 "
            "압력 이상으로 실시한다. (2-1) 최고사용압력이 저압인 배관 및 그 부대설비 "
            "이외의 것으로서 최고사용압력이 30kPa 이하"
        ),
    )
    split_continuation = dict(chunk)
    split_continuation.update(
        chunk_id=20506,
        page=95,
        hierarchy="[FS551] 95페이지",
        content=(
            "KGS FS551 2024 인 것은 시험압력을 최고사용압력으로 할 수 있다. "
            "(2-2) 이미 설치된 사용자공급관은 시험압력을 사용압력 이상으로 할 수 있다."
        ),
    )
    split_result = RagPipeline._tightness_test_pressure_rule(
        "KGS FS551 저압 배관에서 압력이 30 kPa 이하인 경우 기밀시험을 생략할 수 있어?",
        [split_anchor, split_continuation],
    )
    assert split_result is not None
    assert "이 30 kPa 조건만으로 저압 배관의 기밀시험을 생략할 수 없습니다" in split_result[1]
    assert "시험압력을 최고사용압력으로 할 수 있다" in split_result[2]


def test_fs551_tightness_pressure_numeric_classification_covers_multiple_classes():
    definition = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20511,
        "content": "1.3.5 고압 1MPa 이상의 압력. 1.3.6 중압 0.1MPa 이상 1MPa 미만. 1.3.7 저압 0.1MPa 미만.",
    }
    tightness = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20512,
        "hierarchy": "[FS551] 4.2.2.9.3 기밀시험 방법",
        "content": "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다.",
    }
    result = RagPipeline._fs551_tightness_pressure_numeric_classification(
        "KGS FS551 최고사용압력 0.08MPa, 0.1MPa, 2MPa인 경우 기밀시험 압력과 고압·중압·저압 분류를 각각 계산해줘.",
        [tightness],
        [definition],
    )

    assert result is not None
    _definition, _definition_excerpt, _tightness, _tightness_excerpt, answer = result
    assert "0.08MPa (80kPa): 저압" in answer
    assert "0.088MPa (88kPa)" in answer
    assert "0.1MPa (100kPa): 중압" in answer
    assert "0.11MPa (110kPa)" in answer
    assert "2MPa (2000kPa): 고압" in answer
    assert "2.2MPa (2200kPa)" in answer


@pytest.mark.asyncio
async def test_fs551_contextual_tightness_pressure_followup_reuses_prior_pressure_value():
    prior_question = "KGS FS551 최고사용압력 0.5MPa인 배관의 기체 내압시험압력을 계산해줘."
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다. "
            "다만, 다음 기준에 해당하는 경우에는 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 "
            "실시하지 않 을수 있다. (2-1) 최고사용압력이 저압인 배관 및 그 부대설비 이외의 것으로서 "
            "최고사용압력이 30kPa 이하인 것은 시험압력을 최고사용압력으로 할 수 있다. "
            "(2-2) 이미 설치된 사용자공급관은 시험압력을 사용압력 이상으로 할 수 있다. (3) 온도 조건."
        ),
    }

    class ContextualPressureDatabase:
        def history(self, _conversation_id):
            return [{"role": "user", "content": prior_question}]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search(self, _query, _limit, doc_codes=None, _domain=None):
            return [chunk] if doc_codes == ["FS551"] else []

        def search_headings(self, _heading, codes, limit=8):
            return [chunk] if "FS551" in codes else []

        def save_exchange(self, *args):
            return 1

    combined_question = f"{prior_question} 그럼 기밀시험 압력은?"
    combined_rule = RagPipeline._tightness_test_pressure_rule(combined_question, [chunk])
    assert combined_rule is not None
    assert "0.5MPa의 1.1배는 0.55MPa" in combined_rule[1]
    assert "다만, 다음 기준에 해당하는 경우" not in combined_rule[1]

    response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="unused",
            service_hub_fast_model="unused",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        ContextualPressureDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(conversation_id="fs551-pressure-followup", message="그럼 기밀시험 압력은?"))

    assert "0.5MPa의 1.1배는 0.55MPa" in response.answer
    assert "기밀시험압력은 550kPa (0.55MPa) 이상" in response.answer
    assert "다만, 다음 기준에 해당하는 경우" not in response.answer
    assert [citation.doc_code for citation in response.citations] == ["FS551"]
    assert response.citations[0].page == 94

    value_only_response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="unused",
            service_hub_fast_model="unused",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        ContextualPressureDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            conversation_id="fs551-pressure-value-only-followup",
            message="그럼 40kPa는?",
        )
    )
    assert "최고사용압력 0.04MPa의 1.1배" in value_only_response.answer
    assert "기밀시험압력은 44kPa (0.044MPa) 이상" in value_only_response.answer
    assert "최고사용압력 0.5MPa" not in value_only_response.answer
    assert value_only_response.citations[0].page == 94


@pytest.mark.asyncio
async def test_fs551_low_pressure_30kpa_omission_answer_cites_separate_omission_clause(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "일반도시가스사업 제조소 및 공급 배관 기준",
            "filename": "FS551.pdf",
            "file_path": str(tmp_path / "FS551.pdf"),
            "file_hash": "fs551-30kpa-omission-test",
            "page_count": 100,
        },
        [
            {
                "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
                "page": 94,
                "content": (
                    "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다. "
                    "다만, 다음 기준에 해당하는 경우에는 높은 압력 이상으로 실시하지 않을 수 있다. "
                    "(2-1) 최고사용압력이 저압인 배관 및 그 부대설비 이외의 것으로서 "
                    "최고사용압력이 30kPa 이하인 것은 시험압력을 최고사용압력으로 할 수 있다. "
                    "(2-2) 다음 조건. (3) 다음 조항."
                ),
                "search_text": "FS551 저압 배관 30kPa 기밀시험압력 예외",
            },
            {
                "hierarchy": "[FS551] 4 검사 기준 > 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
                "page": 94,
                "content": (
                    "4.2.2.9.6 기밀시험을 생략할 수 있는 가스공급시설은 최고사용압력이 0MPa 이하의 것 "
                    "또는 항상 대기로 개방되어 있는 것으로 한다."
                ),
                "search_text": "FS551 기밀시험 생략 0MPa 대기로 개방",
            },
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(service_hub_model="test"), database, ReasonerMustNotRun()
    )

    response = await pipeline.run(
        ChatRequest(
            message="KGS FS551 저압 배관에서 압력이 30 kPa 이하인 경우 기밀시험을 생략할 수 있어?"
        )
    )

    assert "이 30 kPa 조건만으로 저압 배관의 기밀시험을 생략할 수 없습니다" in response.answer
    assert "별도 기밀시험 생략은 최고사용압력이 0 MPa 이하" in response.answer
    assert "이는 30 kPa 시험압력 예외와 별개의 조건입니다" in response.answer
    assert [citation.page for citation in response.citations] == [94, 94]
    assert "30kPa 이하" in response.citations[0].excerpt
    assert "4.2.2.9.6" in response.citations[1].excerpt


def test_tightness_test_pressure_exceptions_include_both_when_asked_generically():
    query = "KGS FS551 기밀시험 압력 예외 조건은?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": (
            "(2) 기본 시험압력은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 값이다. "
            "(2-1) 최고사용압력이 30kPa 이하인 것은 시험압력을 최고사용압력으로 할수 있다. "
            "(2-2) 이미 설치된 사용자공급관은 시험압력을 사용압력 이상으로 할수 있다. "
            "(3) 다음 조항"
        ),
    }

    result = RagPipeline._tightness_test_pressure_rule(query, [chunk])

    assert result is not None
    _source_chunk, answer, _excerpt = result
    assert "30kPa 이하" in answer
    assert "이미 설치된 사용자공급관" in answer
    assert answer.count("[1]") == 3


@pytest.mark.asyncio
async def test_tightness_pressure_comparison_keeps_both_codes_and_clause_scopes():
    query = "KGS FS551과 FU551의 기밀시험 30 kPa 예외는 저압 배관 제외 조건이 동일해?"
    fs_chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9.3 기밀시험",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다. "
            "다만, 다음 기준에 해당하는 경우에는 기본 압력 이상으로 실시하지 않을 수 있다. "
            "(2-1) 최고사용압력이 저압인 배관 및그 부대설비 이외의 것으로서 최고사용압력이 30kPa 이하인 것은 "
            "시험압력을 최고사용압력으로 할수 있다. (2-2) 이미 설치된 사용자공급관은 사용압력 이상으로 한다."
        ),
    }
    fu_chunk = {
        "document_id": 2,
        "doc_type": "CODE",
        "doc_code": "FU551",
        "title": "가스사용시설 기준",
        "chunk_id": 22354,
        "hierarchy": "[FU551] 4.2.2.1.15 기밀시험",
        "page": 36,
        "filename": "FU551.pdf",
        "score": 1.0,
        "content": (
            "(3-2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다. "
            "(3-2-1) 최고사용압력이 저압인 배관 및그 부대설비 이외의 것으로서, 최고사용압력이 30kPa 이하인 것은 "
            "시험압력을 최고사용압력으로 할수 있다. (3-3) 취성 파괴 우려가 없는 온도에서 시험한다."
        ),
    }
    fs_page_94_chunk = {
        **fs_chunk,
        "content": (
            "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력 이상으로 실시한다. "
            "다만, 다음 기준에 해당하는 경우에는 기본 압력 이상으로 실시하지 않을 수 있다. "
            "(2-1) 최고사용압력이 저압인 배관 및그 부대설비 이외의 것으로서 "
            "최고사용압력이 30kPa 이하"
        ),
    }
    fs_page_95_continuation = {
        **fs_chunk,
        "chunk_id": 20505,
        "page": 95,
        "hierarchy": "[FS551] 95페이지",
        "content": (
            "KGS FS551 2024 인 것은 시험압력을 최고사용압력으로 할 수 있다. "
            "(2-2) 이미 설치된 사용자공급관은 시험압력을 사용압력 이상으로 할 수 있다."
        ),
    }

    paged_result = RagPipeline._tightness_pressure_comparison(
        query, [fu_chunk, fs_page_94_chunk, fs_page_95_continuation]
    )
    assert paged_result is not None
    paged_sources, paged_answer = paged_result
    assert [chunk["doc_code"] for chunk, _excerpt in paged_sources] == [
        "FS551", "FS551", "FU551"
    ]
    assert paged_sources[1][0]["page"] == 95
    assert "시험압력을 최고사용압력으로 할 수 있다" in paged_answer
    assert "기술적 의미와 대상 제한이 같습니다" in paged_answer
    assert "[1] [2] [3]" in paged_answer

    result = RagPipeline._tightness_pressure_comparison(query, [fu_chunk, fs_chunk])

    assert result is not None
    rows, answer = result
    assert [chunk["doc_code"] for chunk, _excerpt in rows] == ["FS551", "FU551"]
    assert "FS551 기본 시험압력" in answer and "FU551 기본 시험압력" in answer
    assert "FS551 30 kPa 예외" in answer and "FU551 30 kPa 예외" in answer
    assert "저압인 배관 및 그 부대설비 이외의 것" in answer
    assert "[1] [2]" in answer

    numeric_query = (
        "KGS FS551과 FU551에서 최고사용압력이 0.5MPa인 배관의 기밀시험압력 기준을 비교하고 "
        "각 시험압력을 계산해줘. 적용 조항과 PDF 쪽수도 구분해줘."
    )
    numeric_result = RagPipeline._tightness_pressure_comparison(
        numeric_query, [fu_chunk, fs_chunk]
    )
    assert numeric_result is not None
    _numeric_sources, numeric_answer = numeric_result
    assert "FS551과 FU551 모두 기밀시험압력은 550kPa(0.55MPa) 이상" in numeric_answer
    assert "입력값 500kPa에는 해당하지 않습니다" in numeric_answer
    assert "다만, 다음 기준에 해당하는 경우" not in numeric_answer

    class PressureComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FS551", "FU551"]
            assert limit == 100
            return [fu_chunk, fs_page_94_chunk]

        def search(self, query, limit, doc_codes, domain):
            assert query == "시험압력을 최고사용압력으로 할 수 있다"
            assert limit == 20 and doc_codes == ["FS551"] and domain == "CODE"
            return [fs_page_95_continuation]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        PressureComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "기술적 의미와 대상 제한이 같습니다" in response.answer
    assert "시험압력을 최고사용압력으로 할 수 있다" in response.answer
    assert [citation.doc_code for citation in response.citations] == [
        "FS551", "FS551", "FU551"
    ]
    assert [citation.page for citation in response.citations] == [94, 95, 36]
    assert response.citations[1].hierarchy.endswith("30 kPa 예외 (계속)")
    assert len(response.citations) == 3

    numeric_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        PressureComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=numeric_query))
    assert "500kPa의 1.1배는 550kPa" in numeric_response.answer
    assert "550kPa(0.55MPa) 이상" in numeric_response.answer
    assert "입력값 500kPa에는 해당하지 않습니다" in numeric_response.answer
    assert [citation.doc_code for citation in numeric_response.citations] == ["FS551", "FU551"]

    class ContextualPressureComparisonDatabase(PressureComparisonDatabase):
        def history(self, _conversation_id):
            return [{"role": "user", "content": query}]

    contextual_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualPressureComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            conversation_id="fs-fu-pressure-followup",
            message=(
                "그럼 두 기준에서 최고사용압력 25 kPa인 저압 배관은 "
                "각각 최소 몇 kPa로 시험해야 해?"
            ),
        )
    )
    assert "25kPa의 1.1배는 27.5kPa" in contextual_response.answer
    assert "저압 배관이므로 두 기준의 30kPa 예외에서는 제외됩니다" in contextual_response.answer
    assert "시험압력 27.5kPa 이상" in contextual_response.answer
    assert [citation.page for citation in contextual_response.citations] == [94, 95, 36]

    class IncompletePressureComparisonDatabase(PressureComparisonDatabase):
        def search(self, _query, _limit, _doc_codes, _domain):
            return []

    incomplete_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        IncompletePressureComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))
    assert "보류" not in incomplete_response.answer
    assert "검색된 원문" in incomplete_response.answer
    assert incomplete_response.citations


def test_tightness_test_acceptance_extractor_covers_temperature_and_pass_criteria():
    query = "KGS FS551 기밀시험의 합격 판정 기준과 시험 온도 조건은 무엇인가?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "content": (
            "(3) 기밀시험은 그 설비가 취성 파괴를 일으킬 우려가 없는 온도에서 실시한다. "
            "(4) 기밀시험은 기밀시험압력에서 누출 등의 이상이 없을 때 합격으로 한다. "
            "(5) 시험 인원은 최소 인원으로 한다."
        ),
    }

    result = RagPipeline._tightness_test_acceptance_rule(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 20504
    assert "취성 파괴를 일으킬 우려가 없는 온도" in answer
    assert "기밀시험압력에서 누출 등의 이상이 없을 때 합격" in answer
    assert "(3)" in excerpt and "(4)" in excerpt


def test_detector_clearance_answer_stays_inside_the_asked_placement_rule():
    query = "가스사용시설의 가스누출경보기는 천장에서 얼마나 떨어져 설치해야 해?"
    rule = (
        "검지부는 천장으로부터 검지부 하단까지의 거리가 0.3m 이하가 되도록 설치한다. "
        "다만, 공기보다 무거운 가스를 사용하는 경우 바닥면으로부터 검지부 상단까지의 거리는 0.3m 이하로 한다."
    )
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22306,
        "hierarchy": "[FU551] 2.8.2.2.3 가스누출자동차단장치 설치 방법",
        "content": f"(1) 검지부 설치 (1-1) {rule} (1-2) 다음 장소에는 검지부를 설치하지 않는다. "
        "(2) 제어부는 가스사용실 연소기 주위에 설치한다.",
    }

    result = RagPipeline._gas_leak_detector_clearance(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 22306
    assert "천장으로부터 검지부 하단" in answer
    assert "공기보다 무거운 가스" in answer
    assert "제어부" not in answer
    assert excerpt == f"(1-1) {rule}"
    assert RagPipeline._gas_leak_detector_clearance(
        "KGS FU551 검지부 수량은 바닥면 둘레 기준으로 어떻게 산정해?", [chunk]
    ) is None


def test_regulator_alarm_answer_extracts_only_threshold_and_time_and_repairs_ocr():
    query = "KGS FU551 정압기실 가스누출경보기는 어느 농도에서, 몇 초 이내에 경보해야 해?"
    clause = (
        "미리 설정된 가스 농도(폭발하한계의 1/4 이하)에서 60초 이내에 "
        "경보를 울리는 것으로 한 다. <개정 09. 9. 25.>"
    )
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22300,
        "hierarchy": "[FU551] 2.8.2.1.1 가스누출경보기 기능",
        "content": (
            "(1) 가스의 누출을 검지하여 농도를 지시한다. "
            f"(2) {clause} "
            "(3) 경보 후에는 계속 경보를 울린다."
        ),
    }

    result = RagPipeline._gas_leak_alarm_threshold(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 22300
    assert "폭발하한계의 1/4 이하" in answer
    assert "60초 이내" in answer
    assert "한 다" not in answer
    assert "<개정" not in answer
    assert "<개정 09. 9. 25.>" in excerpt
    assert "(1) 가스의 누출" not in answer
    assert excerpt.endswith("한 다.") is False


def test_regulator_detector_exclusions_keep_only_the_four_regulator_room_rules():
    query = "KGS FU551 정압기실 가스누출경보기 검지부를 설치하면 안 되는 장소를 모두 알려줘."
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22302,
        "hierarchy": "[FU551] 2.8.2.1.3 가스누출경보기 설치 장소",
        "content": (
            "(2) 다음 기준에 해당하지 않는 곳으로 한다. "
            "(2-1) 증기, 물방울, 기름섞인 연기 등이 직접 접촉될 우려가 있는 곳. "
            "(2-2) 주위 온도 또는 복사열에 의한 온도가 40°C 이상이 되는 곳. "
            "(2-3) 설비 등에 가려져 누출가스의 유통이 원활하지 못한 곳. "
            "(2-4) 차량 및 그 밖의 작업 등으로 인하여 경보기가 파손될 우려가 있는 곳. "
            "(3) 검지부의 설치 높이는 가스의 비중에 맞는 곳으로 한다."
        ),
    }
    unrelated_chunk = {
        **chunk,
        "chunk_id": 22306,
        "hierarchy": "[FU551] 2.8.2.2.3 가스누출자동차단장치 설치 방법",
        "content": "(1-2-1) 출입구 부근 (1-2-2) 환기구 1.5m 이내",
    }

    result = RagPipeline._regulator_detector_forbidden_places(
        query, [chunk, unrelated_chunk]
    )

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 22302
    assert "40°C 이상" in answer
    assert "파손될 우려" in answer
    assert "출입구" not in answer and "환기구" not in answer
    assert answer.count("[1]") == 4
    assert "(2-1)" in excerpt and "(2-4)" in excerpt


def test_regulator_detector_count_applies_the_source_ratio_to_user_perimeter():
    query = (
        "KGS FU551 정압기실의 바닥면 둘레가 45m라면 "
        "가스누출경보기 검지부는 최소 몇 개 설치해야 해? 기준으로 계산해줘."
    )
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22303,
        "hierarchy": "[FU551] 2.8.2.1.4 가스누출경보기 설치 개수",
        "content": (
            "정압기실(지하정압기실을 포함한다)에 설치하는 검지부의 수는 "
            "바닥면 둘레 20m에 1개 이상의 비율로 계산된 수로 한다."
        ),
    }

    result = RagPipeline._regulator_detector_count_for_perimeter(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 22303
    assert "45m ÷ 20m × 1개 = 2.25개" in answer
    assert "산술상 최소 3개" in answer
    assert "원문은 별도 올림 절차를 적지 않습니다" in answer
    assert "20m에 1개 이상" in excerpt

    exact_query = (
        "KGS FU551 정압기실 둘레가 정확히 40m라면 가스누출경보기 검지부는 최소 몇 개 설치해야 하나요?"
    )
    exact_result = RagPipeline._regulator_detector_count_for_perimeter(exact_query, [chunk])
    assert exact_result is not None
    _source_chunk, exact_answer, exact_excerpt = exact_result
    assert "40m ÷ 20m × 1개 = 2개" in exact_answer
    assert "따라서 최소 2개" in exact_answer
    assert "20m에 1개 이상" in exact_excerpt


@pytest.mark.asyncio
async def test_fs551_indoor_detector_count_does_not_mislabel_general_rule_as_regulator_specific():
    query = (
        "KGS FS551 정압기실 바닥면 둘레가 25m라면 가스누출경보기 검지부를 "
        "최소 몇 개 설치해야 해?"
    )
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20410,
        "hierarchy": "[FS551] 2 시설기준 > 13.10 14> > 2.5 배관설비 > 2.5.8 가스누출경보기 > 2.5.8.5.4 건축물 내부 설치",
        "page": 62,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "(4-1-4-1) 가스누출경보기의 검지부의 수는 배관 길이 20m 마다 또는 "
            "바닥면 둘레 20m에 대하여 한개 이상의 비율로 계산한 수 (4-2) 다음 기준"
        ),
    }

    class Fs551IndoorCountDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "2.5.8.5.4"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def search(self, query, limit, codes, domain):
            assert query == "가스누출경보기 검지부 20m"
            assert limit == 100 and codes == ["FS551"] and domain == "CODE"
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        Fs551IndoorCountDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "25m ÷ 20m × 1개 = 1.25개" in response.answer
    assert "산술상 최소 2개" in response.answer
    assert "정압기실 전용 기준이라고 특정하지는 않습니다" in response.answer
    assert response.citations[0].doc_code == "FS551"
    assert response.citations[0].page == 62
    assert "13.10 14>" not in response.citations[0].hierarchy
    assert "2.5.8.5.4(4-1-4-1) 검지부 수량" in response.citations[0].hierarchy


@pytest.mark.asyncio
async def test_fs551_detector_location_and_interval_separates_general_pipe_and_indoor_rules():
    query = "KGS FS551 배관의 가스누출검지기는 어디에 설치하고 설치 간격은 어떻게 정해?"
    rows = [
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 100, "hierarchy": "[FS551] 2.7.2.3.1 검지부 설치장소",
            "page": 64, "filename": "FS551.pdf", "score": 10.0,
            "content": (
                "2.7.2.3.1 검지부 또는 가스누출을 용이하게 검지할 수 있는 구조의 검지구를 "
                "설치하는 장소는 다음과 같다. (1) 긴급차단장치의 부분 (2) 슬리브관·보호관·"
                "방호구조물 등으로 밀폐되어 설치한 배관의 부분 (3) 누출된 가스가 체류하기 쉬운 "
                "구조로 된 배관의 부분"
            ),
        },
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 101, "hierarchy": "[FS551] 2.7.2.4 가스누출검지경보장치 설치개수",
            "page": 64, "filename": "FS551.pdf", "score": 100.0,
                "content": "2.7.2.4에 따라 배관에는 1개 이상의 가스누출검지경보장치를 설치한다.",
        },
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 104, "hierarchy": "[FS551] 2.7.2.3.2 검지부 설치 위치",
            "page": 64, "filename": "FS551.pdf", "score": 10.0,
            "content": (
                "검지부를 설치하는 위치는 조건에 따라 정하되 다음 장소에는 설치하지 않는다. "
                "(1) 증기 등이 직접 접촉할 우려가 있는 곳 (2) 주위 온도나 복사열로 40℃ 이상이 되는 곳 "
                "(3) 누출가스 흐름이 원활하지 못한 곳 (4) 차량·작업으로 파손될 우려가 있는 곳"
            ),
        },
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 105, "hierarchy": "[FS551] 2.7.2.3.3 검지부 설치 높이",
            "page": 64, "filename": "FS551.pdf", "score": 10.0,
            "content": "검지부의 설치높이는 가스비중, 주위상황, 처리설비높이 등의 조건에 따라 정한다.",
        },
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 102, "hierarchy": "[FS551] 2.5.8.5.4 건축물 내부 설치",
            "page": 61, "filename": "FS551.pdf", "score": 100.0,
            "content": (
                "(4) (3)에 따른 환기를 확보할 수 없는 경우에는 다음 기준에 따라 가스누출경보기를 "
                "설치하거나 용접부에 대하여 비파괴시험을 실시하여 이상이 없거나 2중 보호관으로 "
                "설치할 수 있다. (4-1) 가스누출경보기를 설치하는 경우에는 다음 기준에 따른다."
            ),
        },
        {
            "document_id": 1, "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
            "chunk_id": 103, "hierarchy": "[FS551] 2.5.8.5.4 건축물 내부 설치",
            "page": 62, "filename": "FS551.pdf", "score": 100.0,
            "content": (
                "(4-1-3-1) 검지부는 누출한 가스가 체류하기 쉬운 장소에 설치한다. "
                "(4-1-3-2) 가스 성질과 주위 조건에 따라 설치하되 다음 장소에는 설치하지 않는다. "
                "(4-1-3-3) 검지부 설치 높이는 해당 가스비중, 주위 상황 등의 조건에 따라 정한다. "
                "(4-1-4-1) 검지부의 수는 배관 길이 20m마다 또는 바닥면 둘레 20m에 대하여 "
                "한 개 이상의 비율로 계산한 수 (4-2) 이중 보호관"
            ),
        },
    ]

    class Fs551DetectorRulesDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search(self, query, limit, doc_codes, domain):
            assert query == "가스누출경보기 검지부 20m"
            assert limit == 100 and doc_codes == ["FS551"] and domain == "CODE"
            return [rows[5]]

        def search_headings(self, heading, codes, limit=8):
            assert codes == ["FS551"]
            if heading == "2.7.2.3":
                return [rows[0], rows[2], rows[3]]
            if heading == "2.7.2.4":
                return [rows[1]]
            if heading == "2.5.8.5.4":
                return [rows[4]]
            raise AssertionError(f"Unexpected heading query: {heading}")

        def save_exchange(self, *_args):
            return 77

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="test"),
        Fs551DetectorRulesDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "긴급차단장치" in response.answer
    assert "일반 조항은 배관 길이별 고정 간격을 규정하지 않으므로" in response.answer
    assert "모든 배관에 20m마다 1개" in response.answer
    assert "환기기준을 확보할 수 없을 때" in response.answer
    assert "배관 길이 20m마다 또는 바닥면 둘레 20m" in response.answer
    assert [item.page for item in response.citations] == [64, 64, 61, 62]
    assert "2.7.2.3.1-2.7.2.3.3" in response.citations[0].hierarchy
    assert "2.5.8.5.4(4-1-3), (4-1-4)" in response.citations[3].hierarchy


def test_gas_pressure_test_steps_extracts_requested_ramp_and_hold_clauses():
    query = "KGS FS551에서 기체로 내압시험을 할 때 승압 순서와 압력 유지시간은?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20508,
        "hierarchy": "[FS551] 4.2.2.10 내압시험",
        "content": (
            "(5) 내압시험은 최고사용압력의 1.5배 이상으로 하며, 규정 압력을 유지하는 "
            "시간은 5분부터 20분까지를 표 준으로 한다. (6) "
            "내압시험을 공기 등의 기체로 하는 경우에 압력은 일시에 시험압력까지 "
            "승압하지 않아야 하 며, 먼저 상용압력의 50%까지 승압하고 그 후에는 "
            "상용압력의 10%씩 단계적으로 승압하여 내압시험 압력에 달하였을 때 "
            "누출 등의 이상이 없고, 그후 압력을 내려 상용압력으로 하였을 때 "
            "팽창, 누출 등의 이상이 없으면 합격으로 한다."
        ),
    }

    result = RagPipeline._gas_pressure_test_steps(query, [chunk])

    assert result is not None
    source_rows, answer = result
    source_chunk = source_rows[0][0]
    excerpt = " ".join(row[2] for row in source_rows)
    assert source_chunk["chunk_id"] == 20508
    assert len(source_rows) == 1
    assert "50%" in answer and "10%씩" in answer
    assert "5분부터 20분까지를 표준으로 한다" in answer
    assert "승압 순서(4.2.2.10.3(6))" in answer
    assert "규정압력 유지시간(4.2.2.10.3(5))" in answer
    assert "합격 판정(4.2.2.10.3(6))" in answer
    assert answer.index("승압 순서") < answer.index("규정압력 유지시간") < answer.index("합격 판정")
    assert "4.2.2.10.3(6)" in excerpt and "4.2.2.10.3(5)" in excerpt
    assert "일시에" in excerpt and "5분부터 20분까지" in excerpt


def test_gas_pressure_test_steps_preserves_pdf_page_break_and_cites_both_pages():
    query = "KGS FS551 공기로 기체 내압시험을 할 때 승압 순서와 유지시간, 합격판정은?"
    page_98 = {
        "doc_type": "CODE", "doc_code": "FS551", "document_id": 1,
        "title": "배관 기준", "chunk_id": 1,
        "hierarchy": "[FS551] 4.2.2.10.3 내압시험", "page": 98,
        "filename": "FS551.pdf", "score": 5.0,
        "content": (
            "(5) 내압시험은 최고사용압력의 1.5배 이상으로 하며, 규정 압력을 유지하는 "
            "시간은 5분부터 20분까지를 표"
        ),
    }
    page_99 = {
        "doc_type": "CODE", "doc_code": "FS551", "document_id": 1,
        "title": "배관 기준", "chunk_id": 2,
        "hierarchy": "[FS551] 99페이지", "page": 99,
        "filename": "FS551.pdf", "score": 5.0,
        "content": (
            "KGS FS551 2024 88 준으로 한다. (6) 내압시험을 공기 등의 기체로 하는 경우에 압력은 일시에 시험압력까지 "
            "승압하지 않아야 하며, 먼저 상용압력의 50%까지 승압하고 그 후에는 상용압력의 "
            "10%씩 단계적으로 승압하여 내압시험 압력에 달하였을 때 누출 등의 이상이 없고, "
            "그 후 압력을 내려 상용압력으로 하였을 때 팽창, 누출 등의 이상이 없으면 합격으로 한다."
        ),
    }

    result = RagPipeline._gas_pressure_test_steps(query, [page_98, page_99])

    assert result is not None
    source_rows, answer = result
    assert [item[0]["page"] for item in source_rows] == [98, 99]
    assert "50%" in answer and "10%씩" in answer
    assert "5분부터 20분까지" in answer
    assert "유지시간(4.2.2.10.3(5))" in answer
    assert "[1][2]" in answer
    assert "승압 순서(4.2.2.10.3(6))" in answer
    assert "합격 판정(4.2.2.10.3(6))" in answer


def test_gas_pressure_test_conditions_extract_allowed_medium_and_prerequisite_checks():
    query = "KGS FS551에서 기체로 내압시험을 할 수 있는 경우와 시험 전에 필요한 검사는?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20508,
        "hierarchy": "[FS551] 4.2.2.10 내압시험",
        "content": (
            "(1) 내압시험은 수압으로 실시한다. 다만, 중압 이하의 배관, 길이 50m 이하로 "
            "설치되는 고압배관과 부득이한 이유로 물을 채우는 것이 부적당한 경우에는 공기나 "
            "위험성이 없는 불활성기체로 할 수 있다. "
            "(2) 공기 등의 기체 압력으로 시험하는 경우 강관 용접부 전길이에 대하여 "
            "시험 전에 방사선투과시험을 하고 등급 2급(중압 이하 배관은 3급) 이상임을 확인한다. "
            "(3) 중압 이상 강관의 양 끝부에는 동등 이상의 성능이 있는 앤드 캡을 용접 부착하고 "
            "비파괴시험을 실시한 후 내압시험을 한다. (4) 취성파괴 우려가 없는 온도에서 실시한다."
        ),
    }

    result = RagPipeline._gas_pressure_test_conditions(query, [chunk])

    assert result is not None
    source_chunk, answer, excerpt = result
    assert source_chunk["chunk_id"] == 20508
    assert "50m 이하" in answer and "불활성기체" in answer
    assert "전길이" in answer and "방사선투과시험" in answer
    assert "2급(중압 이하 배관은 3급) 이상" in answer
    assert "비파괴시험" in answer
    assert "방사선투과시험" in excerpt


@pytest.mark.asyncio
async def test_gas_tightness_and_pressure_test_rules_are_kept_separate():
    query = (
        "KGS FS551 기밀시험과 내압시험에서 기체를 사용하는 조건이 어떻게 달라? "
        "시험 전 검사도 구분해줘."
    )
    tightness_chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2 검사방법 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.9.1 시공감리 시 누출 여부와 배관 내부 시험가스 방출 여부를 확인한다. "
            "4.2.2.9.2 정기검사 조항. 4.2.2.9.3 배관의 기밀시험 방법은 다음과 같다. "
            "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. "
            "다만, 통과하는 가스로 기밀시험을 할 수 있는 경우는 다음과 같다. "
            "(1-1) 최고사용압력이 고압이나 중압으로 길이가 15m 미만인 배관은 "
            "동일 재료·치수·시공방법으로 하고 최고사용압력의 1.1배 이상에서 누출이 없음을 확인한다. "
            "(1-2) 최고사용압력이 저압인 배관은 정해진 방법으로 시험한다. "
            "(1-3) 기설치된 사용자공급관의 기밀시험을 하는 경우. "
            "(2) 기밀시험압력은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력으로 실시한다. "
            "다만, 저압배관에 해당하는 경우 최고사용압력이 30kPa 이하이면 최고사용압력 이상으로 할 수 있다. "
            "(2-1) 보충 조건. "
            "(3) 설비는 취성파괴 우려가 없는 온도에서 시험한다. "
            "(4) 기밀시험은 기밀시험압력에서 누출 등의 이상이 없을 때 합격으로 한다. (5) 시험 안전조치."
        ),
    }
    pressure_chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20508,
        "hierarchy": "[FS551] 4.2 검사방법 > 4.2.2.10 내압시험",
        "page": 98,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.10.1 중압 이상 배관은 최고사용압력의 1.5배(고압의 가스시설로서 공기·질소 등의 "
            "기체로 내압시험을 실시하는 경우에는 1.25배) 이상의 압력으로 시험한다. "
            "4.2.2.10.2 압력강하 및 이상변형, 파손이 없는지 확인한다. "
            "4.2.2.10.3 도시가스공급시설의 내압시험은 다음 기준에 따라 실시한다. "
            "(1) 내압시험은 수압으로 실시한다. 다만, 중압 이하의 배관, 길이 50m 이하로 "
            "설치되는 고압배관과 부득이한 이유로 물을 채우는 것이 부적당한 경우에는 공기나 "
            "위험성이 없는 불활성기체로 할 수 있다. "
            "(2) 공기 등의 기체 압력으로 시험하는 경우 강관 용접부 전길이에 대하여 "
            "시험 전에 방사선투과시험을 하고 등급 2급(중압 이하 배관은 3급) 이상임을 확인한다. "
            "(3) 중압 이상 강관의 양 끝부에는 동등 이상의 성능이 있는 앤드 캡을 용접 부착하고 "
            "비파괴시험을 실시한 후 내압시험을 한다. (4) 취성파괴 우려가 없는 온도에서 실시한다. "
            "(5) 내압시험은 최고사용압력의 1.5배(고압의 가스시설로서 공기·질소 등의 기체로 내압시험을 "
            "실시하는 경우에는 1.25배) 이상으로 하며, 규정 압력을 유지하는 시간은 5분부터 20분까지를 표준으로 한다. "
            "(6) 기체 내압시험 후 상용압력에서 팽창·누출 등의 이상이 없으면 합격으로 한다. (7) 인원 기준."
        ),
    }

    result = RagPipeline._gas_tightness_vs_pressure_test(
        query, [tightness_chunk], [pressure_chunk]
    )

    assert result is not None
    tightness_source, tightness_excerpt, pressure_source, _pressure_answer, pressure_excerpt, answer = result
    assert tightness_source["page"] == 94
    assert pressure_source["page"] == 98
    assert "15m 미만" in tightness_excerpt
    assert "기밀시험과 내압시험을 서로 다른 조항으로 규정합니다" in answer
    assert "전길이" in answer and "방사선투과시험" in answer
    assert "2급(중압 이하 배관은 3급) 이상" in answer
    assert "50m 이하" in pressure_excerpt
    assert "기밀시험 매체" in answer
    assert "15m 미만" in answer and "50m 이하" in answer
    assert "기체 내압시험의 사전요건" in answer
    assert "통과가스 허용 조건과는 별도" in answer
    assert "압력강하 및 이상변형, 파손이 없는지 확인한다" in pressure_excerpt

    checklist_result = RagPipeline._gas_tightness_vs_pressure_test(
        "KGS FS551의 기밀시험과 내압시험을 현장 작업 순서로 혼동하지 않도록 체크리스트로 만들어줘.",
        [tightness_chunk],
        [pressure_chunk],
    )
    assert checklist_result is not None
    assert "현장 체크리스트는 기밀시험과 내압시험을 별도 절차로 관리합니다" in checklist_result[-1]
    assert "[A] 기밀시험" in checklist_result[-1] and "[B] 내압시험" in checklist_result[-1]

    numeric_query = (
        "KGS FS551 배관 최고사용압력 0.08MPa일 때 기밀시험압력과 "
        "내압시험압력을 계산해줘."
    )
    numeric_comparison = RagPipeline._gas_tightness_vs_pressure_test(
        numeric_query, [tightness_chunk], [pressure_chunk]
    )
    assert numeric_comparison is not None
    numeric_pressure = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        numeric_query,
        [pressure_chunk],
        [{
            "doc_type": "CODE",
            "doc_code": "FS551",
            "chunk_id": 20500,
            "page": 12,
            "content": (
                "1.3.5 고압이란 1MPa 이상의 압력(게이지압력)을 말한다. "
                "1.3.6 중압이란 0.1MPa 이상 1MPa 미만의 압력을 말한다. "
                "1.3.7 저압이란 0.1MPa 미만의 압력을 말한다. 1.3.8 액화가스."
            ),
        }],
    )
    assert numeric_pressure is not None
    assert "저압" in numeric_pressure[-1]
    assert "내압시험압력은 0.08MPa 저압 배관에 적용할 규정 배수가" in numeric_pressure[-1]

    class GasComparisonDatabase:
        definition_chunk = {
            "document_id": 1,
            "doc_type": "CODE",
            "doc_code": "FS551",
            "title": "배관 기준",
            "chunk_id": 20500,
            "hierarchy": "[FS551] 1 일반사항 > 1.3 용어정의",
            "page": 12,
            "filename": "FS551.pdf",
            "score": 1.0,
            "content": (
                "1.3.5 고압이란 1MPa 이상의 압력(게이지압력)을 말한다. "
                "1.3.6 중압이란 0.1MPa 이상 1MPa 미만의 압력을 말한다. "
                "1.3.7 저압이란 0.1MPa 미만의 압력을 말한다. 1.3.8 액화가스."
            ),
        }

        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert codes == ["FS551"]
            assert limit <= 100
            if "기밀시험" in heading:
                return [tightness_chunk]
            if "내압시험" in heading:
                return [pressure_chunk]
            if "용어정의" in heading:
                return [self.definition_chunk]
            return []

        def save_exchange(self, *args):
            return 1

        def search(self, _query, _limit, doc_codes=None, _domain=None):
            return [pressure_chunk] if doc_codes == ["FS551"] else []

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        GasComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "기밀시험과 내압시험을 서로 다른 조항으로 규정합니다" in response.answer
    assert "기밀시험 매체: 공기 또는 위험성이 없는 불활성기체" in response.answer
    assert "강관 용접부 전길이" in response.answer
    assert [citation.page for citation in response.citations] == [94, 98]
    assert "15m 미만" in response.citations[0].excerpt
    assert "50m 이하" in response.citations[1].excerpt

    purpose_medium_pass_query = (
        "KGS FS551 기밀시험과 내압시험은 시험 목적·매체·합격 판정이 어떻게 달라?"
    )
    purpose_medium_pass_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        GasComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=purpose_medium_pass_query))
    assert "- 목적:" in purpose_medium_pass_response.answer
    assert "기밀시험은 시험압력에서 누출 등 이상이 없어야 합니다" in purpose_medium_pass_response.answer
    assert "압력강하 및 이상변형, 파손이 없는지 확인한다" in purpose_medium_pass_response.answer
    assert "시험매체" in purpose_medium_pass_response.answer
    assert "15m 미만" not in purpose_medium_pass_response.answer
    assert "방사선투과시험" not in purpose_medium_pass_response.answer
    assert [citation.page for citation in purpose_medium_pass_response.citations] == [94, 98]

    detailed_query = (
        "KGS FS551 길이 70m, 최고사용압력 0.5MPa인 배관에서 물 채우기가 곤란해 "
        "기밀시험을 하려는 경우 시험가스와 압력 기준을 알려줘. "
        "50m 이하 예외와 물 채우기 곤란 예외를 섞지 말고, 어느 조건을 충족해야 하는지와 "
        "조항·PDF 쪽수를 구분해줘."
    )
    assert RagPipeline._fs551_gas_pressure_test_numeric_classification(
        detailed_query + " 내압시험 기체 시험매체 조건 비교",
        [pressure_chunk],
        [GasComparisonDatabase.definition_chunk],
    ) is not None
    detailed_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        GasComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=detailed_query))

    assert "기밀시험 압력은 FS551 4.2.2.9.3(2)" in detailed_response.answer
    assert "0.5MPa를 대입하면 1.1 × 0.5 = 0.55MPa" in detailed_response.answer
    assert "70m는 4.2.2.9.3(1-1)" in detailed_response.answer
    assert "30kPa 이하에 관한 시험압력 대체 규정은 입력값 0.5MPa(500kPa)에는 적용되지 않습니다" in detailed_response.answer
    assert "{pressure_text}" not in detailed_response.answer
    assert "50m·물 충전 부적당 조건은 기밀시험의 예외가 아니라 내압시험" in detailed_response.answer
    assert "중압이므로 내압시험의 ‘중압 이하’ 사유가 적용" in detailed_response.answer
    assert "길이만으로 그 사유가 배제되지는 않습니다" in detailed_response.answer
    assert [citation.page for citation in detailed_response.citations] == [94, 12, 98]
    assert "4.2.2.9.3(2)" in detailed_response.citations[0].excerpt
    assert "1.3.5" in detailed_response.citations[1].excerpt
    assert "4.2.2.10.3" in detailed_response.citations[2].excerpt

    simple_scope = RagPipeline._gas_pressure_test_high_pressure_exception(
        "KGS FS551에서 70m 고압배관의 내압시험을 할 때 물을 채우기 곤란하면 기체시험이 가능한가?",
        [pressure_chunk],
    )
    assert simple_scope is not None
    assert "70 m 배관은 50 m 이하 고압배관 사유에는 해당하지 않지만" in simple_scope[1]
    assert "물 충전 부적당 사유에는 별도의 길이 제한" in simple_scope[1]

    numeric_hold_query = (
        "KGS FS551 최고사용압력 0.5MPa인 배관을 공기로 내압시험할 때 시험압력을 계산하고, "
        "1.25배와 1.5배 중 무엇이 적용되는지와 유지시간을 알려줘."
    )
    numeric_hold_response = await RagPipeline(
        SimpleNamespace(
            service_hub_model="unused",
            service_hub_fast_model="unused",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        GasComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=numeric_hold_query))
    assert "0.5MPa × 1.5 = 0.75MPa 이상" in numeric_hold_response.answer
    assert "1.25배는 고압 가스시설" in numeric_hold_response.answer
    assert "이번 중압 배관에는 적용되지 않습니다" in numeric_hold_response.answer
    assert "5~20분이 표준" in numeric_hold_response.answer
    assert [citation.page for citation in numeric_hold_response.citations] == [12, 98]


def test_fs551_numeric_pressure_reassembles_split_definition_and_continuation_chunks():
    query = (
        "KGS FS551 최고사용압력 0.5MPa인 배관을 공기로 내압시험할 때 "
        "시험압력과 1.25배·1.5배 적용 여부, 유지시간을 알려줘."
    )
    base = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "filename": "FS551.pdf",
        "score": 1.0,
    }
    definitions = [
        {
            **base,
            "chunk_id": 1,
            "page": 13,
            "hierarchy": "[FS551] 1.3.5 고압",
            "content": "1.3.5 고압이란 1MPa 이상의 압력을 말한다.",
        },
        {
            **base,
            "chunk_id": 2,
            "page": 13,
            "hierarchy": "[FS551] 1.3.6 중압",
            "content": "1.3.6 중압이란 0.1MPa 이상 1MPa 미만의 압력을 말한다.",
        },
        {
            **base,
            "chunk_id": 3,
            "page": 13,
            "hierarchy": "[FS551] 1.3.7 저압",
            "content": "1.3.7 저압이란 0.1MPa 미만의 압력을 말한다.",
        },
    ]
    pressure = [
        {
            **base,
            "chunk_id": 10,
            "page": 98,
            "hierarchy": "[FS551] 4.2.2.10.1",
            "content": "4.2.2.10.1 중압 이상의 배관은 최고사용압력의 1.5배(고압의 가스시설로서 공기·질소 등의 기체로 내압시험을 실시하는 경우에는 1.25배) 이상의 압력으로 시험한다.",
        },
        {
            **base,
            "chunk_id": 11,
            "page": 98,
            "hierarchy": "[FS551] 4.2.2.10.3",
            "content": "4.2.2.10.3 (1) 내압시험은 수압으로 실시한다. 다만, 중압 이하의 배관, 길이 50m 이하 고압배관과 부득이한 이유로 물을 채우는 것이 부적당한 경우에는 공기나 불활성기체로 할 수 있다. (2) 공기 등의 기체 압력으로 시험하는 경우 강관 용접부 전길이에 방사선투과시험을 한다. (3) 중압 이상 강관은 양 끝부에 앤드캡을 용접 부착하고 비파괴시험 후 내압시험을 한다. (5) 내압시험은 최고사용압력의 1.5배 이상으로 하며 5분부터 20분까지를 표준으로 한다.",
        },
        {
            **base,
            "chunk_id": 12,
            "page": 99,
            "hierarchy": "[FS551] 99페이지",
            "content": "KGS FS551 2024 준으로 한다. (6) 내압시험을 공기 등의 기체로 하는 경우 압력은 상용압력의 50%까지 먼저 승압하고 이후 10%씩 단계적으로 승압한다.",
        },
    ]

    result = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        query, pressure, definitions
    )

    assert result is not None
    assert "0.5MPa는 중압" in result[-1]
    assert "0.5MPa × 1.5 = 0.75MPa 이상" in result[-1]
    assert "1.25배는 고압 가스시설" in result[-1]
    assert "5~20분" in result[-1]

    bare_pressure_result = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        "KGS FS551 0.05MPa 배관을 질소로 내압시험하면 시험압력을 계산해야 해?",
        pressure,
        definitions,
    )
    assert bare_pressure_result is not None
    assert "0.05MPa는 저압" in bare_pressure_result[-1]
    assert "5510.05MPa" not in bare_pressure_result[-1]

    class_only_result = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        "KGS FS551 최고사용압력 1MPa인 배관의 내압시험 등급과 시험압력 배수는?",
        pressure,
        definitions,
    )
    assert class_only_result is not None
    assert "1MPa는 고압" in class_only_result[-1]
    assert "1MPa × 1.5 = 1.5MPa 이상" in class_only_result[-1]
    assert "공기·질소 등 기체로 시험하는 고압 가스시설이면 1.25배" in class_only_result[-1]

    multi_boundary_result = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        "KGS FS551 최고사용압력 0.099MPa, 0.1MPa, 1MPa 각각 압력등급과 내압시험압력을 계산해줘.",
        pressure,
        definitions,
    )
    assert multi_boundary_result is not None
    assert "0.099MPa:" in multi_boundary_result[-1]
    assert "0.099MPa는 저압" in multi_boundary_result[-1]
    assert "0.1MPa:" in multi_boundary_result[-1]
    assert "0.1MPa는 중압" in multi_boundary_result[-1]
    assert "0.1MPa × 1.5 = 0.15MPa 이상" in multi_boundary_result[-1]
    assert "1MPa:" in multi_boundary_result[-1]
    assert "1MPa는 고압" in multi_boundary_result[-1]
    assert "1MPa × 1.5 = 1.5MPa 이상" in multi_boundary_result[-1]

    contextual_numeric_result = RagPipeline._fs551_gas_pressure_test_numeric_classification(
        "KGS FS551 최고사용압력 2MPa인 배관을 질소로 내압시험할 때 시험압력은? "
        "KGS FS551 최고사용압력 0.5MPa인 배관을 질소로 내압시험할 때 시험압력과 유지시간을 알려줘.",
        pressure,
        definitions,
    )
    assert contextual_numeric_result is not None
    assert "2MPa는 고압" in contextual_numeric_result[-1]
    assert "2MPa × 1.25 = 2.5MPa 이상" in contextual_numeric_result[-1]


def test_fs551_long_distance_hold_time_handles_volume_boundaries_and_hold_time_wording():
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 905,
        "page": 96,
        "hierarchy": "[FS551] 4.2.2.9.4(5) 장거리 구간 기밀유지시간",
        "content": (
            "4.2.2.9.4(5) 내용적 300m3 이상 5000m3 미만은 48시간, "
            "5000m3 이상 10000m3 미만은 96시간, "
            "10000m3 이상 25000m3 미만은 120시간, 25000m3 이상은 144시간."
        ),
    }
    result = RagPipeline._fs551_long_distance_tightness_hold_time(
        "KGS FS551 장거리 배관 내용적이 299m³, 300m³, 4,999m³, 5,000m³, "
        "9,999m³, 10,000m³, 24,999m³, 25,000m³일 때 기밀유지시간을 각각 계산해줘.",
        [chunk],
    )
    assert result is not None
    answer = result[1]
    assert "V=299m³: 300m³ 미만" in answer
    assert "V=300m³: 48시간" in answer
    assert "V=4999m³: 48시간" in answer
    assert "V=5000m³: 96시간" in answer
    assert "V=9999m³: 96시간" in answer
    assert "V=10000m³: 120시간" in answer
    assert "V=24999m³: 120시간" in answer
    assert "V=25000m³: 144시간" in answer


@pytest.mark.asyncio
async def test_fs551_long_distance_hold_time_followup_uses_prior_user_code():
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 906,
        "page": 96,
        "hierarchy": "[FS551] 4.2.2.9.4(5) 장거리 구간 기밀유지시간",
        "filename": "FS551.pdf",
        "content": (
            "4.2.2.9.4(5) 내용적 300m3 이상 5000m3 미만은 48시간, "
            "5000m3 이상 10000m3 미만은 96시간, "
            "10000m3 이상 25000m3 미만은 120시간, 25000m3 이상은 144시간."
        ),
        "score": 100.0,
    }

    class FollowupDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return set(codes)

        def search_headings(self, heading, codes, limit=100):
            assert heading == "기밀시험"
            assert codes == ["FS551"]
            return [chunk]

        def search_pages(self, pages, codes, limit=100):
            assert pages == [96]
            assert codes == ["FS551"]
            return []

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FollowupDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 5,000m³이면 몇 시간 유지해야 해?",
            history=[
                ChatTurn(
                    role="user",
                    content="KGS FS551 장거리 배관 내용적이 300m³ 이상일 때 기밀유지시간 표를 설명해줘.",
                )
            ],
        )
    )
    assert "V=5000m³: 96시간" in response.answer
    assert response.citations[0].doc_code == "FS551"
    assert response.citations[0].page == 96


def test_gas_pressure_test_water_fill_exception_is_not_given_the_50m_limit():
    query = (
        "KGS FS551 고압 배관이 길이 50m를 초과해도 물을 채우기 부적당하면 "
        "기체 내압시험을 할 수 있어? 그 경우도 50m 제한이 붙어?"
    )
    context = "user: KGS FS551에서 기체로 내압시험을 할 수 있는 조건과 시험 전 검사는?"
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20508,
        "hierarchy": "[FS551] 4.2 검사방법 > 4.2.2.10 내압시험",
        "content": (
            "(1) 내압시험은 수압으로 실시한다. 다만, 중압 이하의 배관, 길이 50m 이하로 "
            "설치되는 고압배관과 부득이한 이유로 물을 채우기 부적당한 경우에는 공기나 "
            "위험성이 없는 불활성기체로 할 수 있다. "
            "(2) 공기 등의 기체 압력으로 시험하는 경우 강관 용접부 전길이에 대하여 "
            "시험 전에 방사선투과시험을 하고 등급 2급(중압 이하 배관은 3급) 이상임을 확인한다. "
            "(3) 중압 이상 강관의 양 끝부에는 배관용 앤드 캡(END CAP), 막음플랜지를 용접 부착하고 "
            "비파괴시험을 실시한 후 내압시험을 한다."
        ),
    }

    result = RagPipeline._gas_pressure_test_high_pressure_exception(
        query, [chunk], context
    )

    assert result is not None
    source, answer, excerpt = result
    assert source["chunk_id"] == 20508
    assert "세 갈래" in answer
    assert "50 m 제한은 길이 50 m 이하 고압배관 사유에 붙고" in answer
    assert "별도의 길이 제한이 적혀 있지 않습니다" in answer
    assert "방사선투과시험" in answer and "전길이" in answer
    assert "시험 전 끝단 조치" in answer and "비파괴시험" in answer
    assert "(1)" in excerpt and "(2)" in excerpt and "(3)" in excerpt

    direct_query = (
        "KGS FS551 고압 배관이 70m이고 부득이한 이유로 물을 채우기 부적당해 "
        "공기로 내압시험하려면, 허용 조건과 시험 전 검사 및 길이 제한을 구분해줘."
    )
    direct_result = RagPipeline._gas_pressure_test_high_pressure_exception(
        direct_query, [chunk]
    )
    assert direct_result is not None
    _, direct_answer, _ = direct_result
    assert "70 m 배관은 50 m 이하 고압배관 사유에는 해당하지 않지만" in direct_answer
    assert "별도 사유로 기체시험을 검토할 수 있으며" in direct_answer
    assert "전길이" in direct_answer and "방사선투과시험" in direct_answer
    assert RagPipeline._gas_pressure_test_high_pressure_exception(
        "고압 배관이 50m를 넘더라도 부득이해서 물을 채우기 부적당하면 "
        "기체 내압시험을 할 수 있어? 그 경우도 50m 제한이 붙어?",
        [chunk],
        "관계없는 다른 문서 대화",
    ) is None


@pytest.mark.asyncio
async def test_fs551_exactly_15m_does_not_satisfy_passthrough_gas_exception():
    query = "FS551에서 최고사용압력이 중압이고 배관 길이가 정확히 15m이면 통과가스로 기밀시험할 수 있나요?"
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. "
            "다만, 통과하는 가스로 기밀시험을 할 수 있는 경우는 다음과 같다. "
            "(1-1) 최고사용압력이 고압이나 중압으로 길이가 15m 미만인 배관 또는 그 부대설비로서 "
            "그 이음부와 동일재료, 동일치수 및 동일시공방법에 따르고 최고사용압력의 1.1배 이상인 "
            "압력에서 누출이 없는가를 확인하고 정해진 방법으로 기밀시험을 한 경우 "
            "(1-2) 최고사용압력이 저압인 배관 또는 그 부대설비로서 정해진 방법으로 시험한 경우 "
            "(1-3) 기설치된 사용자공급관의 기밀시험을 하는 경우. (2) 다음 항목"
        ),
    }

    class TightnessGasDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert "기밀시험" in heading
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        TightnessGasDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "15m 배관은 이 예외에 해당하지 않습니다" in response.answer
    assert "기설치 사용자공급관" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 94
    assert "15m 미만" in response.citations[0].excerpt

    under_limit_query = "FS551에서 최고사용압력이 중압이고 길이가 14m인 배관은 통과가스로 기밀시험해도 돼?"
    under_limit_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        TightnessGasDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=under_limit_query))
    assert "14m는 (1-1)의 길이 조건인 15m 미만을 충족합니다" in under_limit_response.answer
    assert "동일 재료·치수·시공방법" in under_limit_response.answer
    assert "최고사용압력의 1.1배 이상" in under_limit_response.answer
    assert "4.2.2.9.4(1) 또는 (2)의 방법" in under_limit_response.answer

    boundary_result = RagPipeline._fs551_tightness_passthrough_gas_rules(
        "KGS FS551 신규 고압·중압 배관 길이 14.9m, 15m, 15.1m일 때 "
        "통과가스 기밀시험 허용 여부와 압력 조건을 각각 정리해줘.",
        [chunk],
    )
    assert boundary_result is not None
    boundary_answer = boundary_result[1]
    assert boundary_answer.count("15m 미만을 충족합니다") == 1
    assert "14.9m" in boundary_answer
    assert "15m 배관은 이 예외에 해당하지 않습니다" in boundary_answer
    assert "15.1m 배관은 이 예외에 해당하지 않습니다" in boundary_answer

    hydrogen_test_gas_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        TightnessGasDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message=(
                "KGS FS551 기밀시험에서 기본 시험가스는 무엇이고 "
                "수소를 시험가스로 사용할 수 있는 조건은?"
            )
        )
    )
    assert "공기 또는 위험성이 없는 불활성기체가 원칙" in hydrogen_test_gas_response.answer
    assert "통과가스는 다음 세 경우에만 허용" in hydrogen_test_gas_response.answer
    assert "15m 미만" in hydrogen_test_gas_response.answer
    assert "저압인 배관" in hydrogen_test_gas_response.answer
    assert "기설치된 사용자공급관" in hydrogen_test_gas_response.answer
    assert "별도의 일반 시험가스로 임의 투입하는 허용이 아니라" in hydrogen_test_gas_response.answer
    assert hydrogen_test_gas_response.citations[0].page == 94

    class ContextualTightnessGasDatabase(TightnessGasDatabase):
        def history(self, _conversation_id):
            return [
                {
                    "role": "user",
                    "content": "KGS FS551 중압 배관에 통과가스 기밀시험을 할 수 있는 조건은?",
                },
                {
                    "role": "assistant",
                    "content": "중압 배관은 15m 미만 등 (1-1)의 조건에 해당해야 합니다.",
                },
            ]

    contextual_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualTightnessGasDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 정확히 15m면?",
            conversation_id="tightness-gas-followup",
        )
    )
    assert "15m 배관은 이 예외에 해당하지 않습니다" in contextual_response.answer
    assert len(contextual_response.citations) == 1
    assert contextual_response.citations[0].page == 94

    contextual_under_limit_response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ContextualTightnessGasDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그럼 같은 배관 길이가 14.9m면 조건을 만족해? 이음부와 누출 확인 조건도 필요한가?",
            conversation_id="tightness-gas-followup-under-limit",
        )
    )
    assert "14.9m는 (1-1)의 길이 조건인 15m 미만을 충족합니다" in contextual_under_limit_response.answer
    assert "동일 재료·치수·시공방법" in contextual_under_limit_response.answer
    assert "최고사용압력의 1.1배 이상" in contextual_under_limit_response.answer
    assert "4.2.2.9.4(1) 또는 (2)의 방법" in contextual_under_limit_response.answer
    assert contextual_under_limit_response.citations[0].page == 94


@pytest.mark.asyncio
async def test_fs551_passthrough_gas_buried_followup_keeps_12h_method_scope():
    clause = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20510,
        "hierarchy": "[FS551] 4.2.2.9.3 기밀시험",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "(1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. "
            "다만, 통과하는 가스로 기밀시험을 할 수 있는 경우는 다음과 같다. "
            "(1-1) 최고사용압력이 고압이나 중압으로 길이가 15m 미만인 배관으로서 "
            "4.2.2.9.4(1)이나 4.2.2.9.4(2)에 따른 방법으로 기밀시험을 한 경우 "
            "(1-2) 최고사용압력이 저압인 배관으로서 정해진 방법으로 시험한 경우 "
            "(1-3) 기설치된 사용자공급관의 기밀시험을 하는 경우. (2) 시험압력 기준"
        ),
    }
    method = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20511,
        "hierarchy": "[FS551] 4.2.2.9.4 신규 배관",
        "page": 95,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.9.4 신규 배관 (1) 발포액을 사용한다. "
            "(2) 시험가스 농도가 0.2% 이하에서 작동하는 가스검지기를 사용한다. "
            "매설된 배관은 시험가스를 넣어서 12시간 경과한 후 판정한다. "
            "(3) 통과가스 방법은 매설배관을 24시간 경과한 후 판정한다."
        ),
    }

    class BuriedFollowupDatabase:
        def history(self, _conversation_id):
            return [
                {
                    "role": "user",
                    "content": "KGS FS551의 통과가스로 기밀시험을 할 수 있는 조건을 정리해줘.",
                },
                {
                    "role": "assistant",
                    "content": "중압 배관은 15m 미만 등 조건을 충족해야 합니다.",
                },
            ]

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험"
            assert codes == ["FS551"]
            assert limit == 100
            return [clause]

        def search_pages(self, pages, codes, limit=8):
            assert pages == [95]
            assert codes == ["FS551"]
            assert limit == 100
            return [method]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        BuriedFollowupDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message="그 경우 매설이면 몇 시간 후 판정해?",
            conversation_id="fs551-buried-followup",
        )
    )

    assert response.intent == "fact"
    assert "12시간 경과 후 판정" in response.answer
    assert "24시간은" in response.answer
    assert response.citations[0].doc_code == "FS551"
    assert response.citations[0].page == 95


@pytest.mark.asyncio
async def test_fs551_supervision_and_regular_inspection_comparison_stays_on_requested_clauses():
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "일반도시가스사업 배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.9 기밀시험 또는 누출검사",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": (
            "4.2.2.9.1 시공감리를 하는 때에는 압력유지시간 등을 고려하여 시험을 실시하여 누출 여부를 확인하고, "
            "배관내부의 시험가스의 방출 여부를 확인한다. "
            "4.2.2.9.2 정기검사를 하는 때에는 기밀시험을 실시(기밀시험 시기가 도래한 경우에만 한다)하고, "
            "그 밖에 가스누출검지기를 이용하여 가스누출여부를 확인하여 이상이 있는 지하매설 배관에 "
            "대해서는 보링작업에 의한 누출검사를 실시한다. "
            "4.2.2.9.3 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다."
        ),
    }

    class FS551InspectionComparisonDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "기밀시험 또는 누출검사"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        FS551InspectionComparisonDatabase(),
        ReasonerMustNotRun(),
    ).run(
        ChatRequest(
            message=(
                "KGS FS551에서 시공감리와 정기검사 때 기밀시험·누출검사를 어떻게 구분하는지 "
                "원문 조항을 인용해 요약해줘. 원문에 없는 검사 의무는 덧붙이지 말아줘."
            )
        )
    )

    assert response.intent == "comparison"
    assert "시공감리(4.2.2.9.1)" in response.answer
    assert "정기검사(4.2.2.9.2)" in response.answer
    assert "기밀시험 시기가 도래한 경우에만" in response.answer
    assert "가스누출검지기" in response.answer and "보링작업" in response.answer
    assert "최고사용압력" not in response.answer
    assert "불활성기체" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 94
    assert "4.2.2.9.1" in response.citations[0].excerpt
    assert "4.2.2.9.2" in response.citations[0].excerpt
    assert "4.2.2.9.3" not in response.citations[0].excerpt


def test_fs551_manual_valve_check_count_applies_greater_of_fixed_minimum_and_rate():
    query = (
        "KGS FS551 전체 매몰형 밸브가 150개이고 박스형 밸브도 150개라면, "
        "작동 확인은 각각 최소 몇 개 해야 해? 기준의 비율대로 계산해줘."
    )
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "chunk_id": 20502,
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.7 가스차단장치",
        "content": (
            "4.2.2.7.3 가스차단장치의 작동상태(수동식 밸브만 한다)는 개폐조작에 의하여 다음과 같이 확인한다. "
            "(1) 매몰형 밸브는 20개소 또는 전체 매몰형 밸브 설치수량의 20% 중 많은 수 이상 "
            "(2) 박스형 밸브는 20개소 또는 전체 박스형 밸브 설치수량의 50% 중 많은 수 이상 "
            "(3) (1) 및 (2)의 방법으로 확인하지 않은 가스차단장치는 도시가스사업자가 실시한 자체점검 기록"
        ),
    }

    result = RagPipeline._fs551_manual_shutoff_check_count(query, [chunk])

    assert result is not None
    source, answer, excerpt = result
    assert source["chunk_id"] == 20502
    assert "매몰형 밸브(총 150개): 150 × 20% = 30개소" in answer
    assert "박스형 밸브(총 150개): 150 × 50% = 75개소" in answer
    assert "큰 값인 최소 30개소" in answer and "큰 값인 최소 75개소" in answer
    assert "수동식 가스차단장치" in answer
    assert "개폐조작" in excerpt and "20%" in excerpt and "50%" in excerpt
    assert "<개정" not in excerpt and "<신설" not in excerpt

    fractional = RagPipeline._fs551_manual_shutoff_check_count(
        "KGS FS551 전체 매몰형 밸브 101개와 박스형 밸브 41개면 최소 몇 개 확인해?",
        [chunk],
    )
    assert fractional is not None
    assert "101 × 20% = 20.2개소 → 정수 개소로 올림 21개소" in fractional[1]
    assert "41 × 50% = 20.5개소 → 정수 개소로 올림 21개소" in fractional[1]

    boundary_values = RagPipeline._fs551_manual_shutoff_check_count(
        "KGS FS551 매몰형 밸브 149개와 150개일 때 각각 최소 몇 개소 확인해?",
        [chunk],
    )
    assert boundary_values is not None
    assert "매몰형 밸브(총 149개)" in boundary_values[1]
    assert "매몰형 밸브(총 150개)" in boundary_values[1]

    minimum_floor = RagPipeline._fs551_manual_shutoff_check_count(
        "KGS FS551 매몰형 밸브 50개와 박스형 밸브 30개면 최소 몇 개 확인해?",
        [chunk],
    )
    assert minimum_floor is not None
    assert "50 × 20% = 10개소; 20개소와 비교해 큰 값인 최소 20개소" in minimum_floor[1]
    assert "30 × 50% = 15개소; 20개소와 비교해 큰 값인 최소 20개소" in minimum_floor[1]


@pytest.mark.asyncio
async def test_fs551_manual_valve_count_route_answers_without_llm():
    query = "KGS FS551 전체 매몰형 밸브가 150개이고 박스형 밸브도 150개라면 각각 최소 몇 개 확인해?"
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.7 가스차단장치",
        "page": 93,
        "filename": "FS551.pdf",
        "chunk_id": 20502,
        "content": (
            "4.2.2.7.3 가스차단장치의 작동상태(수동식 밸브만 한다)는 개폐조작에 의하여 다음과 같이 확인한다. "
            "(1) 매몰형 밸브는 20개소 또는 전체 매몰형 밸브 설치수량의 20% 중 많은 수 이상 "
            "(2) 박스형 밸브는 20개소 또는 전체 박스형 밸브 설치수량의 50% 중 많은 수 이상 "
            "(3) (1) 및 (2)의 방법으로 확인하지 않은 가스차단장치는 도시가스사업자가 실시한 자체점검 기록"
        ),
        "score": 1.0,
    }

    class ValveDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "가스차단장치"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ValveDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "최소 30개소" in response.answer and "최소 75개소" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 93
    assert "20%" in response.citations[0].excerpt
    assert "50%" in response.citations[0].excerpt


@pytest.mark.asyncio
async def test_fs551_manual_valve_count_route_recognizes_plain_language_type_names():
    query = (
        "KGS FS551 지하매설 수동차단밸브 101개소와 박스 안 밸브 41개소를 "
        "표본검사한다면 각각 몇 개소를 검사해야 해?"
    )
    chunk = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "hierarchy": "[FS551] 4 검사 기준 > 4.2.2.7 가스차단장치",
        "page": 93,
        "filename": "FS551.pdf",
        "chunk_id": 20502,
        "content": (
            "4.2.2.7.3 가스차단장치의 작동상태(수동식 밸브만 한다)는 개폐조작에 의하여 다음과 같이 확인한다. "
            "(1) 매몰형 밸브는 20개소 또는 전체 매몰형 밸브 설치수량의 20% 중 많은 수 이상 "
            "(2) 박스형 밸브는 20개소 또는 전체 박스형 밸브 설치수량의 50% 중 많은 수 이상 "
            "(3) (1) 및 (2)의 방법으로 확인하지 않은 가스차단장치는 도시가스사업자가 실시한 자체점검 기록"
        ),
        "score": 1.0,
    }

    class ValveDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "가스차단장치"
            assert codes == ["FS551"]
            assert limit == 100
            return [chunk]

        def save_exchange(self, *args):
            return 1

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        ValveDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert "매몰형 밸브(총 101개): 101 × 20% = 20.2개소 → 정수 개소로 올림 21개소" in response.answer
    assert "박스형 밸브(총 41개): 41 × 50% = 20.5개소 → 정수 개소로 올림 21개소" in response.answer
    assert response.citations[0].page == 93


@pytest.mark.asyncio
async def test_fs551_manual_valve_count_route_joins_page_continuation_rows():
    query = "KGS FS551 매몰형 밸브가 150개라면 최소 몇 개소를 확인해?"
    intro = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20503,
        "hierarchy": "[FS551] 4.2.2.7.3 가스차단장치 작동상태",
        "page": 93,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": "4.2.2.7.3 가스차단장치의 작동상태(수동식 밸브만 한다)는 개폐조작에 의하여 확인한다.",
    }
    rows = {
        "document_id": 1,
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "배관 기준",
        "chunk_id": 20504,
        "hierarchy": "[FS551] 94페이지",
        "page": 94,
        "filename": "FS551.pdf",
        "score": 1.0,
        "content": "(1) 매몰형 밸브는 20개소 또는 전체 매몰형 밸브 설치수량의 20% 중 많은 수 이상 (2) 박스형 밸브는 20개소 또는 전체 박스형 밸브 설치수량의 50% 중 많은 수 이상",
    }

    class SplitValveDatabase:
        def history(self, _conversation_id):
            return []

        def existing_document_codes(self, codes):
            return {code for code in codes if code == "FS551"}

        def search_headings(self, heading, codes, limit=8):
            assert heading == "가스차단장치"
            assert codes == ["FS551"]
            assert limit == 100
            return [intro]

        def search_pages(self, pages, codes, limit=8):
            assert pages == [94]
            assert codes == ["FS551"]
            assert limit == 100
            return [rows]

        def save_exchange(self, *args):
            return 3

    response = await RagPipeline(
        SimpleNamespace(service_hub_model="unused"),
        SplitValveDatabase(),
        ReasonerMustNotRun(),
    ).run(ChatRequest(message=query))

    assert response.intent == "fact"
    assert "최소 30개소" in response.answer
    assert response.citations[0].chunk_id == 20504
    assert response.citations[0].page == 94


def test_shutoff_device_answer_keeps_four_exceptions_under_their_parent_rule():
    query = (
        "KGS FU551에서 가스누출경보차단장치를 설치해야 하는 시설과 "
        "설치하지 않을 수 있는 예외를 구분해줘."
    )
    chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22304,
        "hierarchy": "[FU551] 2.8.2.2.1 가스누출자동차단장치 설치 대상",
        "content": (
            "특정가스사용시설 및 영업장 면적 100m2 이상 또는 지하 시설에는 장치를 설치한다. "
            "다만, 다음 중 어느 하나에 해당하는 경우에는 장치를 설치하지 않을 수 있다. "
            "(1) 월 사용 예정량 2000m3 미만이고 각 배관에 퓨즈콕 등이 설치되며 각 연소기에 "
            "소화안전장치가 부착된 경우 "
            "(2) 불시 차단으로 재해 및 손실이 막대할 우려가 있는 시설로서 "
            "2.8.2.2.3(4)에서 규정하는 경우와 그 시설의 산업용 가스보일러 "
            "<개정 11. 1. 3.> "
            "(3) 연동차단기능의 다기능 가스안전계량기를 설치하는 경우 "
            "(4) 가정용 연소기가 설치된 시설 <신설 23. 8. 25.>"
        ),
    }

    result = RagPipeline._automatic_shutoff_install_rules(query, [chunk])

    assert result is not None
    _source_chunk, answer, excerpt = result
    assert answer.count(" [1]") == 6
    assert "설치하지 않을 수 있다." in answer
    assert "2.8.2.2.3(4)" in answer
    assert "다기능 가스안전계량기" in answer
    assert "<개정" not in answer and "<신설" not in answer
    assert "2.8.2.2.3(4)" in excerpt
    assert "<개정 11. 1. 3.>" in excerpt and "<신설 23. 8. 25.>" in excerpt


@pytest.mark.asyncio
async def test_fu671_detector_install_exceptions_are_not_confused_with_fu551(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-install-exceptions",
            "page_count": 80,
        },
        [
            {
                "hierarchy": "[FU671] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치",
                "page": 68,
                "content": (
                    "수소연료사용시설에는 가스가 누출될 경우 이를 신속히 검지하여 효과적으로 대응할 수 있도록 "
                    "다음 기준에 따라 가스누출검지경보장치를 설치한다."
                ),
                "search_text": "FU671 수소연료사용시설 검지경보장치 설치",
            },
            {
                "hierarchy": (
                    "[FU671] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치 > "
                    "2.8.2.3.1 사업소 안"
                ),
                "page": 70,
                "content": (
                    "(1) 건축물 안 압축기·수소생산설비·수소저장설비 주위의 체류 우려 장소는 "
                    "설비군 바닥면 둘레 10m마다 1개 이상. (2) 조건에 해당하는 건축물 밖 장소는 "
                    "설비군 바닥면 둘레 20m마다 1개 이상."
                ),
                "search_text": "FU671 수소생산설비 검지경보장치 10m 20m 설치 개수",
            },
            {
                "hierarchy": (
                    "[FU671] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치 > "
                    "2.8.2.3.2 사업소 밖"
                ),
                "page": 70,
                "content": (
                    "2.8.2.3.2 사업소 밖 (1) 긴급차단장치 설치 부분 (2) 밀폐·매설 구간 "
                    "(3) 가스 체류 우려 장소. 2.8.2.3.3 검출부는 천정부터 하단까지 0.3m 이하. "
                    "2.8.2.3.4 고천장에서는 누출되기 쉬운 수소설비 상부에 포집갓을 설치하며 "
                    "0.4m 이상으로 한다. 2.8.2.3.5 경보부는 관계자 상주 장소에 설치한다."
                ),
                "search_text": "FU671 검출부 설치장소 천정 0.3m 포집갓 경보부",
            },
        ],
    )
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU551",
            "title": "도시가스 사용시설 기준",
            "filename": "FU551.pdf",
            "file_path": str(tmp_path / "FU551.pdf"),
            "file_hash": "fu551-shutoff-exceptions",
            "page_count": 100,
        },
        [
            {
                "hierarchy": "[FU551] 2.8.2.2.1 가스누출자동차단장치 설치 대상",
                "page": 72,
                "content": (
                    "특정가스사용시설 및 영업장 면적 100m2 이상 또는 지하 시설에는 장치를 설치한다. "
                    "다만, 다음 중 어느 하나에 해당하는 경우에는 가스누출경보차단장치나 "
                    "가스누출자동차단기를 설치하지 않을 수 있다. "
                    "(1) 월 사용 예정량 2000m3 미만이고 각 배관에 퓨즈콕 등이 설치되며 각 연소기에 "
                    "소화안전장치가 부착된 경우 (2) 불시 차단으로 재해 및 손실이 막대할 우려가 있는 "
                    "시설로서 정해진 경우 및 그 시설의 산업용 가스보일러 "
                    "(3) 연동차단기능의 다기능 가스안전계량기를 설치하는 경우 "
                    "(4) 가정용 연소기가 설치된 시설"
                ),
                "search_text": "FU551 설치 대상 가스누출자동차단장치 2000m3 예외 가스안전계량기",
            }
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(
            service_hub_model="test",
            service_hub_fast_model="test-fast",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        database,
        ReasonerMustNotRun(),
    )

    response = await pipeline.run(
        ChatRequest(
            message=(
                "KGS FU671 수소 생산시설에서 가스누출경보차단장치를 설치해야 하는 조건과 "
                "설치하지 않아도 되는 예외를 근거 조항별로 정리해줘."
            )
        )
    )

    assert "FU671" in response.answer
    assert "일반 면제 목록이 없습니다" in response.answer
    assert "FU551" in response.answer
    assert "이를 FU671에 그대로 적용할 수는 없습니다" in response.answer
    assert "월 사용 예정량 2000m3 미만" in response.answer
    assert "다기능 가스안전계량기" in response.answer
    assert "가정용 연소기" in response.answer
    assert "천정으로부터 하단까지 0.3m 이하" in response.answer
    assert [item.doc_code for item in response.citations] == ["FU671", "FU671", "FU671", "FU551"]
    assert [item.page for item in response.citations] == [68, 70, 70, 72]


@pytest.mark.asyncio
async def test_fu671_fu551_alarm_limits_and_shutoff_exceptions_are_compared_separately(
    tmp_path: Path,
):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU671",
            "title": "수소연료사용시설 기준",
            "filename": "FU671.pdf",
            "file_path": str(tmp_path / "FU671.pdf"),
            "file_hash": "fu671-alarm-comparison",
            "page_count": 80,
        },
        [
            {
                "hierarchy": "[FU671] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치",
                "page": 68,
                "content": (
                    "수소연료사용시설에는 가스가 누출될 경우 신속히 검지할 수 있도록 "
                    "가스누출검지경보장치를 설치한다."
                ),
                "search_text": "FU671 수소연료사용시설 가스누출검지경보장치 설치",
            },
            {
                "hierarchy": (
                    "[FU671] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치 > "
                    "2.8.2.1 가스누출경보기 및 가스누출자동차단장치 기능"
                ),
                "page": 68,
                "content": (
                    "2.8.2.1.2 경보농도는 폭발하한계의 1/4 이하로 한다. "
                    "2.8.2.1.3 경보기의 정밀도는 경보 설정치의 ±25% 이하로 한다. "
                    "2.8.2.1.4 검지부터 발신까지 걸리는 시간은 경보농도의 1.6배 농도에서 "
                    "보통 30초 이내로 한다. 2.8.2.1.5 후속 조건이다."
                ),
                "search_text": "FU671 경보농도 폭발하한계 1/4 1.6배 30초",
            },
        ],
    )
    database.replace_document(
        {
            "doc_type": "CODE",
            "doc_code": "FU551",
            "title": "도시가스 사용시설 기준",
            "filename": "FU551.pdf",
            "file_path": str(tmp_path / "FU551.pdf"),
            "file_hash": "fu551-alarm-comparison",
            "page_count": 100,
        },
        [
            {
                "hierarchy": (
                    "[FU551] 2.8.2 가스누출경보기 및 가스누출자동차단장치 설치 > "
                    "2.8.2.1 가스누출경보기 설치 > 2.8.2.1.1 경보기 성능"
                ),
                "page": 71,
                "content": (
                    "경보농도는 폭발하한계의 1/4 이하로 하고 그 농도에서 "
                    "60초 이내에 경보를 발하도록 한다."
                ),
                "search_text": "FU551 경보기 폭발하한계 1/4 60초",
            },
            {
                "hierarchy": "[FU551] 2.8.2.2.1 가스누출자동차단장치 설치 대상",
                "page": 72,
                "content": (
                    "특정가스사용시설 및 영업장 면적 100m2 이상 또는 지하 시설에는 장치를 설치한다. "
                    "다만, 다음 중 어느 하나에 해당하는 경우에는 가스누출경보차단장치나 "
                    "가스누출자동차단기를 설치하지 않을 수 있다. "
                    "(1) 월 사용 예정량 2000m3 미만이고 퓨즈콕 등이 설치된 경우 "
                    "(2) 불시 차단으로 손실이 큰 시설 "
                    "(3) 연동차단기능의 다기능 가스안전계량기를 설치한 경우 "
                    "(4) 가정용 연소기가 설치된 시설"
                ),
                "search_text": "FU551 자동차단장치 설치 예외 가정용 연소기",
            },
        ],
    )
    pipeline = RagPipeline(
        SimpleNamespace(
            service_hub_model="test",
            service_hub_fast_model="test-fast",
            retrieval_limit=32,
            context_limit=24,
            max_context_chars=42000,
        ),
        database,
        ReasonerMustNotRun(),
    )

    query = (
        "KGS FU671과 FU551에서 가스누출경보기의 경보농도 기준과 자동 차단장치 "
        "설치 면제 예외를 비교해줘. 서로 다른 규정은 섞지 말고 조항별로 근거를 보여줘."
    )
    codes = extract_document_codes(query)
    assert set(codes) == {"FU671", "FU551"}
    alarm_chunks = database.search_headings("가스누출경보기", ["FU671", "FU551"], limit=100)
    fu671_install_chunks = database.search_headings(
        "가스누출경보기 및 가스누출자동차단장치 설치", ["FU671"], limit=100
    )
    fu551_shutoff_chunks = database.search_headings(
        "가스누출자동차단장치 설치 대상", ["FU551"], limit=100
    )
    assert any(
        item["doc_code"] == "FU671" and "2.8.2.1.2" in item["content"] and "1.6배" in item["content"]
        for item in alarm_chunks
    )
    assert any(
        item["doc_code"] == "FU551"
        and "60초 이내" in item["content"]
        and "폭발하한계의 1/4 이하" in item["content"]
        for item in alarm_chunks
    )
    assert RagPipeline._automatic_shutoff_install_rules(
        "FU551 가스누출경보차단장치 설치 대상과 설치하지 않을 수 있는 예외",
        fu551_shutoff_chunks,
    ) is not None
    assert RagPipeline._fu671_fu551_alarm_and_shutoff_comparison(
        query,
        codes,
        alarm_chunks,
        fu671_install_chunks,
        fu551_shutoff_chunks,
    ) is not None

    response = await pipeline.run(ChatRequest(message=query))

    assert response.intent == "comparison"
    assert "모두 경보기 농도 상한을 폭발하한계(LEL)의 1/4 이하" in response.answer
    assert "FU671에서는" in response.answer and "FU551 정압기실 경보기는" in response.answer
    assert "보통 30초 이내" in response.answer and "60초 이내" in response.answer
    assert "FU551 2.8.2.2.1" in response.answer
    assert "FU671에 그대로 옮겨 적용할 수 없습니다" in response.answer
    assert "단독 수소제조시설의 적용 여부" in response.answer
    assert [item.doc_code for item in response.citations] == ["FU671", "FU551", "FU551", "FU671"]
    assert [item.page for item in response.citations] == [69, 71, 72, 68]
    assert "1.6배 농도" in response.citations[0].excerpt
    assert "가스누출경보차단장치" in response.citations[2].excerpt


def test_numbered_exception_followup_resolves_referenced_clause_and_actions():
    query = "그럼 예외 2에서 말한 설치 제외 장소와 필요한 조치는 뭐야?"
    history = [
        {
            "role": "assistant",
            "content": "예외 2는 2.8.2.2.3(4)에서 정하는 경우를 말합니다.",
        }
    ]
    intro_chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22306,
        "hierarchy": "[FU551] 2.8.2.2.3 설치 방법",
        "content": (
            "2.8.2.2.1(2)에 따라 불시 차단이 위험하거나 가스누출자동차단장치를 설치하여도 "
            "그 설치 목적을 달성할 수 없는 시설은 "
            "다음 2.8.2.2.3(4-1)과 2.8.2.2.3(4-2)의 시설로 하되 "
            "2.8.2.2.3(4-3)에서 정하는 조치를 한다."
        ),
    }
    details_chunk = {
        "doc_type": "CODE",
        "doc_code": "FU551",
        "chunk_id": 22307,
        "hierarchy": "[FU551] 2.8.2.2.3 설치 방법",
        "content": (
            "(4-1) 불시 차단으로 피해 우려가 있는 시설 (4-1-1) 건조로 "
            "(4-2) 장치를 설치해도 목적을 달성할 수 없는 시설 (4-2-1) 개방된 공장 "
            "(4-3) 설치 제외 대상에는 다음 조치를 한다. (4-3-1) 외부 또는 가까운 내부 배관부에 "
            "가스 공급을 쉽게 차단할 수 있는 장치를 설치한다."
        ),
    }

    result = RagPipeline._shutoff_exception_details(
        query, history, [intro_chunk, details_chunk]
    )

    assert result is not None
    cited_rows, answer = result
    assert [row[0]["chunk_id"] for row in cited_rows] == [22306, 22307]
    assert "설치 제외 장소와 필요한 조치" in answer
    assert "(4-3)" in answer and "차단할 수 있는 장치" in answer


def test_fs551_pressure_test_comparison_covers_method_pressure_time_and_acceptance():
    query = (
        "KGS FS551에서 내압시험과 기밀시험의 차이를 시험매체, 시험압력·유지시간, "
        "합격기준으로 비교해줘. 기준에 없는 항목은 추측하지 말고 없다고 표시해."
    )
    chunks = [
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 1, "page": 94,
            "content": (
                "4.2.2.9.3 (1) 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시한다. "
                "(2) 기밀시험은 최고사용압력의 1.1배 또는 8.4kPa 중 높은 압력이상으로 실시한다."
            ),
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 2, "page": 95,
            "content": "(2-1) 특정 조건에서는 최고사용압력으로 할 수 있다. (4) 기밀시험압력에서 누출 등의 이상이 없을 때 합격으로 한다.",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 3, "page": 95,
            "content": "4.2.2.9.4 압력측정기구 종류와 시험부 용적 및 최고사용압력에 따라 정한 기밀유지시간이상 유지한다.",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 4, "page": 96,
            "content": "48×V분(다만, 2880분을 초과한 경우는 2880분으로 할 수 있다)",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 5, "page": 96,
            "content": "자기압력기록계는 최소 기밀 유지시간을 30분으로 하고, 전기식다이어프램형압력계는 최소 기밀 유지시간을 4분으로 한다.",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 6, "page": 98,
            "content": "4.2.2.10.1 중압 이상의 배관은 최고사용압력의 1.5배(고압의 가스시설로서 공기·질소 등의 기체로 내압시험을 실시하는 경우에는 1.25배) 이상으로 한다.",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 7, "page": 98,
            "content": "4.2.2.10.2 압력강하 및 이상변형, 파손이 없는지 확인한다.",
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 8, "page": 98,
            "content": (
                "도시가스공급시설의 내압시험은 다음 기준에 따라 실시한다. (1) 내압시험은 수압으로 실시한다. "
                "(5) 규정 압력을 유지하는 시간은 5분부터 20분까지를 표"
            ),
        },
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 9, "page": 99,
            "content": (
                "KGS FS551 2024 준으로 한다. (6) 내압시험을 공기 등의 기체로 하는 경우에 압력은 일시에 시험압력까지 승압하지 않아야 하며, "
                "먼저 상용압력의 50%까지 승압하고 그 후에는 상용압력의 10%씩 단계적으로 승압하여 내압시험 압력에 달하였을 때 누출 등의 이상이 없고, "
                "그 후 압력을 내려 상용압력으로 하였을 때 팽창, 누출 등의 이상이 없으면 합격으로 한다."
            ),
        },
    ]

    result = RagPipeline._fs551_pressure_test_comparison(query, chunks)

    assert result is not None
    source_rows, answer = result
    assert [row[0]["page"] for row in source_rows] == [94, 95, 96, 98, 99]
    assert "최고사용압력의 1.1배" in answer and "8.4kPa" in answer
    assert "5~20분" in answer and "공통 시간이 정해져 있지 않습니다" in answer
    assert "30분 또는 4분" in answer
    assert "압력강하·이상변형·파손" in answer
    assert "상용압력으로 낮춘 뒤에도 팽창·누출" in answer
    assert "서로 대체할 수 있다는 규정이 없으므로" in answer
    selection_result = RagPipeline._fs551_pressure_test_comparison(
        "KGS FS551 기준에서 기밀시험과 내압시험 중 어떤 시험을 어떤 상황에서 선택하는지 판단해줘.",
        chunks,
    )
    assert selection_result is not None
    assert "하나만 고르는 일반 분기" in selection_result[1]
    natural_result = RagPipeline._fs551_pressure_test_comparison(
        "KGS FS551에서 기밀시험과 내압시험은 시험매체, 압력 단계, 합격 판정이 어떻게 다른가?",
        chunks,
    )
    assert natural_result is not None
    assert "시험매체" in natural_result[1]
    assert "5~20분" in natural_result[1]
    assert "압력강하·이상변형·파손" in natural_result[1]


def test_fs551_pressure_test_comparison_does_not_route_non_comparison_questions():
    chunks = [
        {
            "doc_type": "CODE", "doc_code": "FS551", "chunk_id": 1, "page": 94,
            "content": "4.2.2.9.3 기밀시험은 공기 또는 위험성이 없는 불활성기체로 실시하고 8.4kPa 이상으로 한다.",
        }
    ]

    assert RagPipeline._fs551_pressure_test_comparison(
        "KGS FS551 기밀시험 압력은?", chunks
    ) is None
