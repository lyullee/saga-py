"""Build a clean SAGA database from every unique PDF and the legacy structured sections."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

from saga.database import Database
from saga.indexer import PdfIndexer
from saga.legacy_corpus import load_legacy_export, match_corpus, source_pdfs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uploads", type=Path, default=ROOT / "saga-uploads")
    parser.add_argument("--legacy", type=Path, default=ROOT / "build" / "legacy-lucene-export.tsv")
    parser.add_argument("--target", type=Path, default=ROOT / "data" / "saga-rebuilt.db")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "corpus-rebuild.json")
    args = parser.parse_args()

    target = args.target.resolve()
    data_root = (ROOT / "data").resolve()
    if target.parent != data_root or target.suffix.lower() != ".db":
        raise SystemExit(f"Refusing to replace an unexpected target: {target}")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{target}{suffix}")
        if candidate.exists():
            candidate.unlink()

    sources, duplicate_groups = source_pdfs(list(args.uploads.glob("*.pdf")))
    legacy_documents = load_legacy_export(args.legacy)
    matches, unmatched_sources, unmatched_legacy = match_corpus(sources, legacy_documents)
    if unmatched_legacy:
        names = ", ".join(item.raw_code for item in unmatched_legacy)
        raise SystemExit(f"Legacy documents were not matched: {names}")

    database = Database(target)
    database.initialize()
    indexer = PdfIndexer(database)
    matched_detail: list[dict] = []
    total = len(matches) + len(unmatched_sources)
    current = 0
    for match in matches:
        current += 1
        source = match.source
        legacy = match.legacy
        doc_code = source.filename_code or legacy.doc_code
        with pymupdf.open(source.path) as document:
            page_count = len(document)
        chunks = indexer.chunks_from_segments(
            doc_code,
            ((item.hierarchy, item.page, item.content) for item in legacy.segments),
        )
        metadata = {
            "doc_type": legacy.doc_type,
            "doc_code": doc_code,
            "title": legacy.title,
            "filename": source.path.name,
            "file_path": str(source.path.resolve()),
            "file_hash": source.file_hash,
            "page_count": page_count,
        }
        document_id = database.replace_document(metadata, chunks)
        matched_detail.append(
            {
                "filename": source.path.name,
                "document_id": document_id,
                "doc_code": doc_code,
                "legacy_code": legacy.raw_code,
                "score": round(match.score, 3),
                "chunks": len(chunks),
            }
        )
        if current % 25 == 0:
            print(f"Structured import {current}/{total}", flush=True)

    fallback_detail: list[dict] = []
    for source in unmatched_sources:
        current += 1
        result = indexer.index_pdf(source.path, force=True)
        fallback_detail.append(
            {
                "filename": source.path.name,
                "status": result.status,
                "document_id": result.document_id,
                "chunks": result.chunks,
                "error": result.error,
            }
        )
        print(f"Fresh parser {current}/{total}: {source.path.name} -> {result.status}", flush=True)

    stats = database.stats()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_pdfs": len(list(args.uploads.glob("*.pdf"))),
        "unique_pdfs": len(sources),
        "duplicate_groups": duplicate_groups,
        "legacy_documents": len(legacy_documents),
        "structured_matches": len(matches),
        "fresh_parser_documents": len(unmatched_sources),
        "database": stats,
        "matched": matched_detail,
        "fallback": fallback_detail,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_pdfs", "unique_pdfs", "legacy_documents", "structured_matches",
        "fresh_parser_documents", "database"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
