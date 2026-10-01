from saga.text import extract_code, extract_codes, extract_document_codes, fts_query, search_tokens
from saga.indexer import PdfIndexer
from pathlib import Path


def test_extract_code_normalizes_kgs_code():
    assert extract_code("KGS fs 551 배관 기준") == "FS551"


def test_extract_codes_preserves_order_and_removes_duplicates():
    assert extract_codes("FF551 대신 KGS fs 551, 다시 FF551") == ["FF551", "FS551"]


def test_extract_document_codes_includes_explicit_rule_number_but_not_year():
    assert extract_document_codes("내부통제규정 11900과 KGS FS551 비교") == ["FS551", "11900"]
    assert extract_document_codes("2025년 개정 내용을 알려줘") == []


def test_extract_rule_number():
    assert extract_code("11100 경영관리규정") == "11100"


def test_korean_ngram_search_tokens():
    tokens = search_tokens("저장탱크 이격거리")
    assert "저장탱크" in tokens
    assert "저장" in tokens
    assert "이격거리" in tokens
    assert "이격" in tokens
    assert " OR " in fts_query("저장탱크 이격거리")


def test_metadata_does_not_treat_rule_reference_as_document_code():
    doc_type, code, _ = PdfIndexer.infer_metadata(
        Path("2206-21_융착기_성능확인_지침.pdf"),
        "이 지침은 KGS FS551 2.5.5.8.7에 따른다.",
    )
    assert doc_type == "RULE"
    assert code == "2206-21"


def test_metadata_uses_code_from_code_filename():
    doc_type, code, title = PdfIndexer.infer_metadata(
        Path("FS551_241128.pdf"),
        "일반도시가스사업 제조소 및 공급소 밖의 배관의\n시설ㆍ기술ㆍ검사ㆍ정밀안전진단 기준\nKGS FS551 2024",
    )
    assert doc_type == "CODE"
    assert code == "FS551"
    assert "일반도시가스사업" in title
