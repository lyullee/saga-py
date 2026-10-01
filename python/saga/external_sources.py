from __future__ import annotations

"""Public operational and incident source ingestion for SAGA.

The source files are deliberately kept beside the SQLite database so each
answer can point back to the downloaded snapshot.  The indexer stores NREL's
official composite-data report as a PDF and converts JRC's HIAD workbook into
self-contained event chunks while retaining the original XLSX.
"""

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .database import Database
from .indexer import PdfIndexer
from .text import normalize_text


# The former NREL host redirects inconsistently on Windows; the official NLR
# mirror serves the same OSTI report and is stable for unattended downloads.
NREL_REPORT_URL = "https://docs.nlr.gov/docs/fy23osti/86247.pdf"
HIAD_XLSX_URL = (
    "https://minerva.jrc.ec.europa.eu/en/shorturl/capri/"
    "hiad_22_export_for_users_2026_01_01xlsx"
)
HIAD_LANDING_URL = "https://minerva.jrc.ec.europa.eu/en/shorturl/capri/hiadpt"


# The public datasets are mostly English while operators ask questions in
# Korean.  These high-confidence aliases are applied only after the source
# has been isolated, so a word such as ``압축기`` cannot pull an unrelated KGS
# paragraph into a standards answer.
EXTERNAL_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "압축기": ("compressor", "compression"),
    "고장": ("failure", "fault", "breakdown"),
    "고장률": ("failure rate", "failure"),
    "고장모드": ("failure mode",),
    "유지보수": ("maintenance", "service"),
    "누출": ("leak", "leakage", "release"),
    "누설": ("leak", "leakage", "release"),
    "화재": ("fire", "fires"),
    "폭발": ("explosion", "blast"),
    "충전기": ("dispenser",),
    "디스펜서": ("dispenser",),
    "노즐": ("nozzle",),
    "저장탱크": ("storage tank", "storage"),
    "저장용기": ("storage vessel", "storage"),
    "배관": ("pipeline", "piping"),
    "밸브": ("valve",),
    "원인": ("cause", "root cause"),
    "근본원인": ("root cause", "cause"),
    "결과": ("consequence", "outcome", "damage"),
    "부상": ("injury", "injuries"),
    "손상": ("damage",),
    "교훈": ("lesson", "lessons learned"),
    "예방": ("prevention", "preventive"),
    "대응": ("response", "emergency response", "action"),
    "충전량": ("dispensed", "dispensed hydrogen"),
    "공급량": ("dispensed", "dispensed hydrogen"),
    "충전횟수": ("fills", "fueling events"),
    "충전 횟수": ("fills", "fueling events"),
    "충전 속도": ("fueling rate", "fill rate"),
    "충전시간": ("fueling time", "fill time"),
    "충전 시간": ("fueling time", "fill time"),
    "최종압력": ("final pressure", "pressure"),
    "최종 압력": ("final pressure", "pressure"),
    "용량이용률": ("capacity utilization", "utilization"),
    "용량 이용률": ("capacity utilization", "utilization"),
    "충전소 배치": ("deployment", "stations"),
    "불순물": ("impurities", "SAE J2719"),
    "품질": ("hydrogen quality", "impurities"),
    "충전시간": ("fueling time", "fill time"),
    "이용률": ("utilization", "capacity"),
    "처리량": ("throughput", "capacity"),
    "안전사고": ("safety report", "incident"),
}


def expand_external_query(value: str, source_type: str | None = None) -> str:
    """Add English dataset vocabulary without changing the user's wording."""
    normalized = normalize_text(value)
    lower = normalized.lower()
    additions: list[str] = []
    for korean, aliases in EXTERNAL_QUERY_ALIASES.items():
        if korean not in normalized and not any(alias.lower() in lower for alias in aliases):
            continue
        for alias in aliases:
            if alias.lower() not in lower and alias not in additions:
                additions.append(alias)
    # Dataset-specific anchor terms improve recall for short questions while
    # keeping the source filter as the authority for corpus isolation.
    if source_type and source_type.upper() == "NREL" and "hydrogen" not in lower:
        additions.append("hydrogen station")
    if source_type and source_type.upper() == "HIAD" and "hydrogen" not in lower:
        additions.append("hydrogen incident")
    return normalize_text(" ".join([normalized, *additions]))


@dataclass(slots=True)
class SourceSyncResult:
    source: str
    status: str
    filename: str
    records: int = 0
    chunks: int = 0
    document_id: int | None = None
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SAGA-Safety-RAG/2.1 (+public-source-ingestor)",
            "Accept": "application/pdf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
        },
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return destination


def _xml_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _read_xlsx_rows(path: Path, sheet_name: str = "EVENTS") -> list[list[str]]:
    """Read one worksheet from a standard XLSX without pandas/openpyxl."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
          "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for node in root.findall("main:si", ns):
                shared.append(_xml_text("".join(node.itertext())))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("main:sheets/main:sheet", ns)
        first_sheet = next(
            (sheet for sheet in sheets if sheet.attrib.get("name", "").strip().upper() == sheet_name.upper()),
            sheets[0] if sheets else None,
        )
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get(f"{{{ns['rel']}}}id", "")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in rels.findall("pkgrel:Relationship", ns):
            if relation.attrib.get("Id") == rel_id:
                target = relation.attrib.get("Target")
                break
        if not target:
            return []
        worksheet_path = "xl/" + target.lstrip("/")
        worksheet_path = worksheet_path.replace("xl/xl/", "xl/")
        root = ET.fromstring(archive.read(worksheet_path))
        rows: list[list[str]] = []
        for row in root.findall("main:sheetData/main:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("main:c", ns):
                reference = cell.attrib.get("r", "A1")
                letters = re.match(r"([A-Z]+)", reference.upper())
                if not letters:
                    continue
                column = 0
                for char in letters.group(1):
                    column = column * 26 + ord(char) - 64
                column -= 1
                kind = cell.attrib.get("t", "")
                if kind == "inlineStr":
                    value = _xml_text("".join(cell.itertext()))
                else:
                    raw = cell.findtext("main:v", default="", namespaces=ns)
                    if kind == "s" and raw.isdigit() and int(raw) < len(shared):
                        value = shared[int(raw)]
                    else:
                        value = _xml_text(raw)
                values[column] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
        return rows


def _hiad_records(path: Path) -> list[dict[str, Any]]:
    rows = _read_xlsx_rows(path)
    if not rows:
        return []
    header_index = next(
        (index for index, row in enumerate(rows[:10]) if sum(bool(cell) for cell in row) >= 2),
        0,
    )
    headers: list[str] = []
    for index, value in enumerate(rows[header_index]):
        clean = _xml_text(value) or f"field_{index + 1}"
        headers.append(clean[:120])
    records: list[dict[str, Any]] = []
    for row in rows[header_index + 1 :]:
        values = row + [""] * max(0, len(headers) - len(row))
        record = {headers[index]: _xml_text(values[index]) for index in range(len(headers))}
        if any(record.values()):
            records.append(record)
    return records


class ExternalSourceIngestor:
    def __init__(self, database: Database, indexer: PdfIndexer, data_dir: Path):
        self.database = database
        self.indexer = indexer
        self.data_dir = data_dir

    def sync_nrel(self, force: bool = False) -> SourceSyncResult:
        filename = "NREL_CDP_Retail_86247.pdf"
        path = self.data_dir / filename
        try:
            _download(NREL_REPORT_URL, path, force=force)
            result = self.indexer.index_pdf(path, force=force)
            if result.status == "failed":
                return SourceSyncResult("nrel", "failed", filename, error=result.error)
            self.database.update_document_metadata(
                filename,
                doc_type="NREL",
                doc_code="NREL-CDP",
                title="NREL hydrogen station retail composite data products",
                source_url=NREL_REPORT_URL,
            )
            row = self.database.document_by_filename(filename)
            return SourceSyncResult(
                "nrel", result.status, filename, chunks=result.chunks,
                document_id=int(row["id"]) if row else result.document_id,
            )
        except Exception as exc:
            return SourceSyncResult("nrel", "failed", filename, error=str(exc))

    def sync_hiad(self, force: bool = False) -> SourceSyncResult:
        filename = "HIAD_2.2.xlsx"
        path = self.data_dir / filename
        try:
            _download(HIAD_XLSX_URL, path, force=force)
            file_hash = _sha256(path)
            existing = self.database.document_by_filename(filename)
            if existing and str(existing["file_hash"]) == file_hash and not force:
                return SourceSyncResult(
                    "hiad", "unchanged", filename,
                    records=int(existing["page_count"]),
                    chunks=int(existing["page_count"]),
                    document_id=int(existing["id"]),
                )
            records = _hiad_records(path)
            document_id = self.database.replace_source_records(
                doc_type="HIAD",
                doc_code="HIAD-2.2",
                title="European Hydrogen Incidents and Accidents Database HIAD 2.2",
                filename=filename,
                file_path=str(path),
                file_hash=file_hash,
                source_url=HIAD_LANDING_URL,
                records=records,
            )
            row = self.database.document_by_filename(filename)
            return SourceSyncResult(
                "hiad", "indexed", filename, records=len(records), chunks=len(records),
                document_id=int(row["id"]) if row else document_id,
            )
        except Exception as exc:
            return SourceSyncResult("hiad", "failed", filename, error=str(exc))

    def sync(self, sources: list[str], force: bool = False) -> list[dict[str, Any]]:
        results: list[SourceSyncResult] = []
        for source in dict.fromkeys(sources):
            if source == "nrel":
                results.append(self.sync_nrel(force))
            elif source == "hiad":
                results.append(self.sync_hiad(force))
            else:
                results.append(SourceSyncResult(source, "failed", "", error="지원하지 않는 공개 데이터 소스입니다."))
        return [asdict(result) for result in results]
