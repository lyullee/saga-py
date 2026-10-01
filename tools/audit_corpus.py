"""Read every source PDF and compare it with the Python and legacy indexes."""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]


def decode_legacy(value: str) -> str:
    try:
        return base64.b64decode(value).decode("utf-8", errors="replace")
    except Exception:
        return ""


def audit_legacy(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    documents: set[tuple[str, str]] = set()
    segments = 0
    invalid_rows = 0
    replacement_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 7:
                invalid_rows += 1
                continue
            decoded = [decode_legacy(value) for value in columns[:7]]
            doc_type, doc_code = decoded[0], decoded[1]
            documents.add((doc_type, doc_code))
            segments += 1
            if "\ufffd" in decoded[6]:
                replacement_rows += 1
    return {
        "available": True,
        "documents": len(documents),
        "segments": segments,
        "invalid_rows": invalid_rows,
        "replacement_rows": replacement_rows,
        "document_codes": sorted(code for _doc_type, code in documents),
    }


def audit_database(path: Path) -> dict:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        documents = [dict(row) for row in connection.execute("SELECT * FROM documents ORDER BY filename")]
        chunks = [dict(row) for row in connection.execute(
            """SELECT document_id, COUNT(*) AS chunks, COUNT(DISTINCT page) AS covered_pages,
                      SUM(CASE WHEN instr(content, char(65533)) > 0 THEN 1 ELSE 0 END) AS replacement_chunks,
                      SUM(CASE WHEN length(content) < 80 THEN 1 ELSE 0 END) AS very_short_chunks,
                      SUM(CASE WHEN hierarchy LIKE '%페이지' THEN 1 ELSE 0 END) AS generic_hierarchy_chunks,
                      MIN(length(content)) AS min_chars, MAX(length(content)) AS max_chars,
                      ROUND(AVG(length(content)), 1) AS avg_chars
                 FROM chunks GROUP BY document_id"""
        )]
    by_document = {row["document_id"]: row for row in chunks}
    code_counts = Counter(row["doc_code"] for row in documents)
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in documents:
        hash_groups[row["file_hash"]].append(row["filename"])
    return {
        "documents": len(documents),
        "chunks": sum(int(row["chunks"]) for row in chunks),
        "zero_chunk_documents": [row["filename"] for row in documents if row["id"] not in by_document],
        "unknown_code_documents": [row["filename"] for row in documents if row["doc_code"] == "UNKNOWN"],
        "duplicate_codes": {code: count for code, count in code_counts.items() if count > 1},
        "duplicate_hashes": [files for files in hash_groups.values() if len(files) > 1],
        "replacement_chunks": sum(int(row["replacement_chunks"] or 0) for row in chunks),
        "very_short_chunks": sum(int(row["very_short_chunks"] or 0) for row in chunks),
        "generic_hierarchy_chunks": sum(int(row["generic_hierarchy_chunks"] or 0) for row in chunks),
        "documents_detail": documents,
        "chunks_by_document": by_document,
    }


def audit_pdfs(upload_dir: Path) -> dict:
    files = sorted(upload_dir.glob("*.pdf"), key=lambda item: item.name.lower())
    detail: list[dict] = []
    total_pages = total_chars = empty_pages = replacement_pages = low_text_pages = 0
    failures: list[dict[str, str]] = []
    for index, path in enumerate(files, 1):
        item = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "pages": 0,
            "text_chars": 0,
            "empty_pages": 0,
            "low_text_pages": 0,
            "replacement_pages": 0,
            "replacement_chars": 0,
        }
        try:
            with pymupdf.open(path) as document:
                item["pages"] = len(document)
                for page in document:
                    text = page.get_text("text")
                    chars = len(text.strip())
                    replacements = text.count("\ufffd")
                    item["text_chars"] += chars
                    item["replacement_chars"] += replacements
                    if chars == 0:
                        item["empty_pages"] += 1
                    elif chars < 80:
                        item["low_text_pages"] += 1
                    if replacements >= 3:
                        item["replacement_pages"] += 1
        except Exception as exc:
            failures.append({"filename": path.name, "error": str(exc)})
        total_pages += int(item["pages"])
        total_chars += int(item["text_chars"])
        empty_pages += int(item["empty_pages"])
        low_text_pages += int(item["low_text_pages"])
        replacement_pages += int(item["replacement_pages"])
        detail.append(item)
        if index % 25 == 0 or index == len(files):
            print(f"PDF scan {index}/{len(files)}", flush=True)
    worst = sorted(
        detail,
        key=lambda row: (row["replacement_pages"], row["empty_pages"], row["low_text_pages"]),
        reverse=True,
    )
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "pages": total_pages,
        "text_chars": total_chars,
        "empty_pages": empty_pages,
        "low_text_pages": low_text_pages,
        "replacement_pages": replacement_pages,
        "failures": failures,
        "problem_files": [row for row in worst if row["empty_pages"] or row["replacement_pages"] or row["low_text_pages"]],
        "files_detail": detail,
    }


def build_summary(report: dict) -> str:
    pdf = report["pdf"]
    db = report["database"]
    legacy = report["legacy"]
    consistency = report["consistency"]
    lines = [
        "# SAGA 원본 문서 전수 감사",
        "",
        f"생성 시각: {report['generated_at']}",
        "",
        "## 전체 현황",
        "",
        f"- 원본 PDF: {pdf['files']}개 / {pdf['pages']:,}페이지 / {pdf['bytes'] / 1024 / 1024:.1f} MiB",
        f"- 현재 Python 색인: {db['documents']}문서 / {db['chunks']:,}청크",
        f"- 기존 Lucene 내보내기: {legacy.get('documents', 0)}문서 / {legacy.get('segments', 0):,}세그먼트",
        "",
        "## 원본 추출 품질",
        "",
        f"- 완전 빈 페이지: {pdf['empty_pages']:,}",
        f"- 80자 미만 저텍스트 페이지: {pdf['low_text_pages']:,}",
        f"- 깨진 문자(U+FFFD) 3개 이상 페이지: {pdf['replacement_pages']:,}",
        f"- 열기 실패 PDF: {len(pdf['failures'])}",
        "",
        "## 색인 정합성",
        "",
        f"- 원본에는 있으나 DB에 없는 파일: {len(consistency['unindexed_pdfs'])}",
        f"- DB에는 있으나 원본이 없는 파일: {len(consistency['orphan_database_documents'])}",
        f"- 청크가 0개인 문서: {len(db['zero_chunk_documents'])}",
        f"- 문서번호 UNKNOWN: {len(db['unknown_code_documents'])}",
        f"- 깨진 문자가 포함된 청크: {db['replacement_chunks']:,}",
        f"- 일반 페이지명만 계층으로 가진 청크: {db['generic_hierarchy_chunks']:,}",
        "",
        "## 우선 확인할 파일",
        "",
    ]
    for item in pdf["problem_files"][:30]:
        lines.append(
            f"- `{item['filename']}`: {item['pages']}쪽, 빈 페이지 {item['empty_pages']}, "
            f"저텍스트 {item['low_text_pages']}, 깨진 페이지 {item['replacement_pages']}"
        )
    if not pdf["problem_files"]:
        lines.append("- 원본 텍스트 추출 이상 없음")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-dir", type=Path, default=ROOT / "saga-uploads")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "saga.db")
    parser.add_argument("--legacy", type=Path, default=ROOT / "build" / "legacy-lucene-export.tsv")
    parser.add_argument("--json", type=Path, default=ROOT / "data" / "corpus-audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "data" / "corpus-audit.md")
    args = parser.parse_args()

    pdf = audit_pdfs(args.upload_dir)
    database = audit_database(args.database)
    legacy = audit_legacy(args.legacy)
    pdf_names = {row["filename"] for row in pdf["files_detail"]}
    db_names = {row["filename"] for row in database["documents_detail"]}
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pdf": pdf,
        "database": database,
        "legacy": legacy,
        "consistency": {
            "unindexed_pdfs": sorted(pdf_names - db_names),
            "orphan_database_documents": sorted(db_names - pdf_names),
            "legacy_codes_missing_from_python": sorted(
                set(legacy.get("document_codes", []))
                - {row["doc_code"] for row in database["documents_detail"]}
            ),
        },
    }
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(build_summary(report), encoding="utf-8")
    print(build_summary(report))


if __name__ == "__main__":
    main()
