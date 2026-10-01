from __future__ import annotations

"""National Law Information Center API -> local, searchable PDF ingestion.

The law API returns XML.  SAGA deliberately converts each retrieved law into a
local PDF before indexing so citations behave exactly like KGS code/rule
citations: the UI opens a local PDF rather than sending the user to an
external web page.  The API's detail URL is retained only in the sidecar
metadata for traceability and is never used as a chat citation.
"""

import json
import re
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pymupdf


LAW_CATALOG: dict[str, str] = {
    # User-facing abbreviations used in Korean gas-safety work.
    "도법": "도시가스사업법",
    "액법": "액화석유가스의 안전관리 및 사업법",
    "고법": "고압가스 안전관리법",
    "수소법": "수소경제 육성 및 수소 안전관리에 관한 법률",
    # A small, deliberately conservative safety-law baseline.  The catalog is
    # configurable through the API request, so additional laws can be added
    # without changing the parser.
    "산안법": "산업안전보건법",
    "중대재해법": "중대재해 처벌 등에 관한 법률",
    "재난안전법": "재난 및 안전관리 기본법",
    "소방기본법": "소방기본법",
    "위험물안전법": "위험물안전관리법",
    "화재예방법": "화재의 예방 및 안전관리에 관한 법률",
    "소방시설법": "소방시설 설치 및 관리에 관한 법률",
    "전기안전법": "전기안전관리법",
    "화학물질관리법": "화학물질관리법",
    "시설물안전법": "시설물의 안전 및 유지관리에 관한 특별법",
}


@dataclass(slots=True)
class LawDocument:
    name: str
    abbreviation: str
    serial: str = ""
    law_id: str = ""
    effective_date: str = ""
    promulgation_date: str = ""
    ministry: str = ""
    detail_url: str = ""


@dataclass(slots=True)
class LawSection:
    heading: str
    text: str
    kind: str = "article"


@dataclass(slots=True)
class LawPdfResult:
    abbreviation: str
    name: str
    pdf_path: Path
    metadata_path: Path
    serial: str
    effective_date: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].replace(" ", "")


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _find_text(element: ET.Element, *needles: str) -> str:
    for child in element.iter():
        tag = _local_name(child.tag)
        if any(needle in tag for needle in needles):
            value = _element_text(child)
            if value:
                return value
    return ""


def _decode_xml(payload: bytes) -> str:
    for encoding in ("utf-8", "euc-kr", "cp949"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _parse_xml(payload: bytes | str) -> ET.Element:
    text = _decode_xml(payload) if isinstance(payload, bytes) else payload
    root = ET.fromstring(text)
    error = _find_text(root, "결과메시지", "resultMsg", "errorMessage", "errMsg")
    code = _find_text(root, "결과코드", "resultCode", "errorCode", "errCode")
    if code and code not in {"00", "0", "SUCCESS", "success"} and error:
        raise RuntimeError(f"국가법령정보센터 API 오류({code}): {error}")
    return root


def parse_law_search(payload: bytes | str, abbreviation: str, requested_name: str) -> LawDocument:
    """Parse one law-search XML response without depending on a fixed namespace."""
    root = _parse_xml(payload)
    # A search result can contain many laws. Prefer an exact Korean name and
    # otherwise use the first result returned by the center's relevance order.
    result_nodes = [
        node for node in root.iter()
        if _local_name(node.tag) in {"law", "법령", "lawInfo", "법령정보", "item"}
    ]
    candidates = result_nodes or [root]
    selected = next(
        (node for node in candidates if _find_text(node, "법령명한글", "법령명_한글", "lawNameKor") == requested_name),
        candidates[0],
    )
    return LawDocument(
        name=_find_text(selected, "법령명한글", "법령명_한글", "lawNameKor") or requested_name,
        abbreviation=abbreviation,
        serial=_find_text(selected, "법령일련번호", "lawSerialNum", "MST"),
        law_id=_find_text(selected, "법령ID", "lawId", "법령id"),
        effective_date=_find_text(selected, "시행일자", "enforcementDate", "efYd"),
        promulgation_date=_find_text(selected, "공포일자", "promulgationDate", "announceDate"),
        ministry=_find_text(selected, "소관부처", "소관부처명", "ministry"),
        detail_url=_find_text(selected, "법령상세링크", "법령상세URL", "detailLink", "lawUrl"),
    )


def _article_heading(node: ET.Element) -> str:
    number = _find_text(node, "조문번호", "articleNum", "조번호")
    title = _find_text(node, "조문제목", "articleTitle", "조제목")
    if number and not number.startswith("제") and number.isdigit():
        number = f"제{number}조"
    if number:
        return f"{number}{f'({title})' if title else ''}".strip()
    return title or "조문"


def parse_law_sections(payload: bytes | str) -> tuple[str, list[LawSection]]:
    """Extract article/appendix sections while preserving Korean hierarchy."""
    root = _parse_xml(payload)
    title = _find_text(root, "법령명한글", "법령명_한글", "lawNameKor")
    sections: list[LawSection] = []
    article_nodes = [
        node for node in root.iter()
        if _local_name(node.tag) in {"조문단위", "articleUnit", "article"}
    ]
    for node in article_nodes:
        heading = _article_heading(node)
        body_parts: list[str] = []
        for child in node.iter():
            tag = _local_name(child.tag)
            if child is node or any(token in tag for token in ("조문번호", "조문제목", "articleNum", "articleTitle")):
                continue
            value = _element_text(child)
            if not value:
                continue
            # Parent nodes repeat all descendant text. Keep leaf-like entries
            # and avoid adding the same sentence twice.
            if value not in body_parts and not any(value in existing for existing in body_parts):
                body_parts.append(value)
        body = " ".join(body_parts).strip()
        if body:
            sections.append(LawSection(heading, body, "article"))

    appendix_nodes = [
        node for node in root.iter()
        if _local_name(node.tag) in {"별표단위", "별표", "appendix", "appendixUnit"}
    ]
    for node in appendix_nodes:
        number = _find_text(node, "별표번호", "별표명", "appendixNum", "appendixName")
        body = _element_text(node)
        if body and number:
            body = body.replace(number, "", 1).strip()
        if body:
            sections.append(LawSection(number or "별표", body, "appendix"))

    if not sections:
        # API schema revisions should not result in an empty searchable PDF.
        # The fallback is intentionally labelled so a reviewer can spot it.
        text = _element_text(root)
        if text:
            sections.append(LawSection("법령 원문", text, "body"))
    return title, sections


def _safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:120] or "law"


def _font_file() -> str | None:
    candidates = (
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunsl.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    return next((str(path) for path in candidates if path.exists()), None)


def render_law_pdf(law: LawDocument, sections: Iterable[LawSection], output_dir: Path, *, today: str | None = None) -> tuple[Path, Path]:
    """Render an official XML snapshot into a local PDF and sidecar metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = today or date.today().isoformat().replace("-", "")
    pdf_path = output_dir / f"LAW_{_safe_filename(law.name)}_{stamp}.pdf"
    metadata_path = pdf_path.with_suffix(".json")
    fontfile = _font_file()
    document = pymupdf.open()
    try:
        lines: list[str] = [
            law.name,
            "국가법령정보센터 API 본문 스냅샷",
            f"시행일자: {law.effective_date or '확인되지 않음'}",
            f"공포일자: {law.promulgation_date or '확인되지 않음'}",
            f"소관부처: {law.ministry or '확인되지 않음'}",
            f"조회일자: {stamp}",
            "※ 답변의 근거는 이 로컬 PDF의 조문·별표이며, 적용 전 최신 시행본을 다시 확인하세요.",
            "",
        ]
        for section in sections:
            lines.extend([section.heading, section.text, ""])
        wrapped: list[str] = []
        for line in lines:
            if not line:
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(line, width=62, replace_whitespace=False, break_long_words=False) or [""])
        page_lines = 46
        for start in range(0, max(1, len(wrapped)), page_lines):
            page = document.new_page(width=595, height=842)
            block = "\n".join(wrapped[start:start + page_lines])
            rect = pymupdf.Rect(44, 42, 551, 790)
            kwargs = {"fontname": "korean", "fontsize": 9.3, "color": (0.08, 0.08, 0.08)}
            if fontfile:
                kwargs["fontfile"] = fontfile
            try:
                page.insert_textbox(rect, block, **kwargs)
            except Exception:
                page.insert_textbox(rect, block, fontname="helv", fontsize=9.3, color=(0.08, 0.08, 0.08))
            page.insert_text((44, 816), f"{law.name} · {start // page_lines + 1}", fontsize=7, color=(0.35, 0.35, 0.35))
        document.save(pdf_path)
    finally:
        document.close()
    metadata_path.write_text(
        json.dumps({
            "name": law.name,
            "abbreviation": law.abbreviation,
            "serial": law.serial,
            "law_id": law.law_id,
            "effective_date": law.effective_date,
            "promulgation_date": law.promulgation_date,
            "ministry": law.ministry,
            "detail_url": law.detail_url,
            "retrieved_at": stamp,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pdf_path, metadata_path


class LawApiIngestor:
    def __init__(self, oc: str, base_url: str = "https://www.law.go.kr/DRF", timeout: float = 30.0):
        self.oc = oc.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.oc:
            raise ValueError("국가법령정보센터 API 인증값(SAGA_LAW_API_OC)이 비어 있습니다.")

    def _get(self, endpoint: str, params: dict[str, str]) -> bytes:
        query = urlencode({"OC": self.oc, **params})
        request = Request(
            f"{self.base_url}/{endpoint}?{query}",
            headers={"Accept": "application/xml", "User-Agent": "SAGA-law-ingestor/1.0"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def fetch(self, abbreviation: str, name: str | None = None) -> tuple[LawDocument, bytes]:
        requested_name = name or LAW_CATALOG.get(abbreviation, abbreviation)
        search_payload = self._get(
            "lawSearch.do",
            {
                "target": "law",
                "type": "XML",
                "query": requested_name,
                "display": "10",
                "page": "1",
                "sort": "efdes",
            },
        )
        law = parse_law_search(search_payload, abbreviation, requested_name)
        if not law.serial and not law.law_id:
            raise RuntimeError(f"'{requested_name}' 검색 결과에서 법령 식별자를 찾지 못했습니다.")
        identifier = law.serial or law.law_id
        body_payload = self._get(
            "lawService.do",
            {"target": "law", "type": "XML", ("MST" if law.serial else "ID"): identifier},
        )
        return law, body_payload

    def fetch_and_render(self, abbreviation: str, output_dir: Path, *, force: bool = False) -> LawPdfResult:
        name = LAW_CATALOG.get(abbreviation, abbreviation)
        law, body_payload = self.fetch(abbreviation, name)
        stamp = date.today().isoformat().replace("-", "")
        expected = output_dir / f"LAW_{_safe_filename(law.name)}_{stamp}.pdf"
        metadata_path = expected.with_suffix(".json")
        if expected.exists() and metadata_path.exists() and not force:
            return LawPdfResult(abbreviation, law.name, expected, metadata_path, law.serial, law.effective_date)
        title, sections = parse_law_sections(body_payload)
        if title:
            law.name = title
        pdf_path, metadata_path = render_law_pdf(law, sections, output_dir, today=stamp)
        return LawPdfResult(abbreviation, law.name, pdf_path, metadata_path, law.serial, law.effective_date)

    def ingest_catalog(self, abbreviations: Iterable[str], output_dir: Path, *, force: bool = False) -> list[LawPdfResult]:
        results: list[LawPdfResult] = []
        for abbreviation in abbreviations:
            if abbreviation not in LAW_CATALOG:
                raise ValueError(f"지원하지 않는 법령 키입니다: {abbreviation}")
            results.append(self.fetch_and_render(abbreviation, output_dir, force=force))
        return results
