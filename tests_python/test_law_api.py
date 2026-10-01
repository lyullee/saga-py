from pathlib import Path

import pymupdf

from saga.indexer import PdfIndexer
from saga.law_api import (
    LAW_CATALOG,
    LawDocument,
    LawSection,
    parse_law_search,
    parse_law_sections,
    render_law_pdf,
)


SEARCH_XML = """
<lawSearch>
  <law>
    <법령명한글>도시가스사업법</법령명한글>
    <법령일련번호>12345</법령일련번호>
    <법령ID>001234</법령ID>
    <시행일자>20260101</시행일자>
    <법령상세링크>https://www.law.go.kr/법령/도시가스사업법</법령상세링크>
  </law>
</lawSearch>
"""

BODY_XML = """
<법령>
  <기본정보><법령명한글>도시가스사업법</법령명한글></기본정보>
  <조문>
    <조문단위>
      <조문번호>제1조</조문번호><조문제목>목적</조문제목>
      <조문내용>이 법은 도시가스사업을 합리적으로 조정하고 안전을 확보함을 목적으로 한다.</조문내용>
    </조문단위>
    <조문단위>
      <조문번호>제2조</조문번호><조문제목>정의</조문제목>
      <항><항번호>①</항번호><항내용>도시가스란 배관으로 공급되는 가스를 말한다.</항내용></항>
    </조문단위>
  </조문>
  <별표><별표번호>별표 1</별표번호><내용>안전거리 기준</내용></별표>
</법령>
"""


def test_law_search_and_body_parser_preserve_identifiers():
    law = parse_law_search(SEARCH_XML, "도법", "도시가스사업법")
    assert law.name == "도시가스사업법"
    assert law.serial == "12345"
    assert law.effective_date == "20260101"

    title, sections = parse_law_sections(BODY_XML)
    assert title == "도시가스사업법"
    assert sections[0].heading == "제1조(목적)"
    assert "안전을 확보" in sections[0].text
    assert any(section.kind == "appendix" for section in sections)


def test_law_snapshot_is_a_searchable_pdf(tmp_path: Path):
    law = LawDocument(name="도시가스사업법", abbreviation="도법", serial="12345")
    pdf_path, metadata_path = render_law_pdf(
        law,
        [LawSection("제1조(목적)", "도시가스사업의 안전을 확보한다.")],
        tmp_path,
        today="20260922",
    )
    assert pdf_path.name == "LAW_도시가스사업법_20260922.pdf"
    assert metadata_path.exists()
    with pymupdf.open(pdf_path) as document:
        text = "\n".join(page.get_text() for page in document)
    assert "도시가스사업법" in text
    assert "안전을 확보한다" in text


def test_law_pdf_metadata_is_classified_as_law():
    doc_type, code, title = PdfIndexer.infer_metadata(
        Path("LAW_고압가스 안전관리법_20260922.pdf"),
        "고압가스 안전관리법\n제1조(목적)",
    )
    assert doc_type == "LAW"
    assert code == "LAW-고압가스 안전관리법"
    assert title == "고압가스 안전관리법"


def test_catalog_contains_gas_and_safety_laws():
    assert LAW_CATALOG["도법"] == "도시가스사업법"
    assert LAW_CATALOG["액법"] == "액화석유가스의 안전관리 및 사업법"
    assert LAW_CATALOG["고법"] == "고압가스 안전관리법"
    assert LAW_CATALOG["수소법"].startswith("수소경제")
    assert "산안법" in LAW_CATALOG
