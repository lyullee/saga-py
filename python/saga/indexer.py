from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pymupdf

from .database import Database
from .text import build_search_text, extract_code, normalize_text


HEADING_RE = re.compile(
    r"^(?:제\s*\d+(?:-\d+)?\s*(?:장|절|조)(?:의\s*\d+)?(?:\s*[\(（].{1,80}?[\)）])?|"
    r"\d+(?:\.\d+){0,5}\.?\s+.{1,100}|부록\s*\d*.*)$"
)
RULE_ARTICLE_RE = re.compile(
    r"^(?P<number>제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?)"
    r"(?P<title>\s*[\(（][^\)）]{1,100}[\)）])?\s*(?P<body>.*)$"
)
RULE_CHAPTER_RE = re.compile(r"^(?P<number>제\s*\d+\s*(?:장|절))\s*(?P<title>.*)$")
RULE_KIND_RE = re.compile(r"(정관|규칙|규정|강령|요령|지침)")
RULE_NUMBER_RE = re.compile(r"(?<!\d)(\d{4,5}(?:-\d+)?)(?!\d)")


@dataclass(slots=True)
class IndexResult:
    filename: str
    status: str
    document_id: int | None = None
    chunks: int = 0
    error: str | None = None


class PdfIndexer:
    def __init__(self, database: Database, chunk_chars: int = 1800, overlap_chars: int = 180, ocr_page: Callable[[pymupdf.Page], str] | None = None):
        self.database = database
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.ocr_page = ocr_page

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def infer_metadata(path: Path, first_pages: str) -> tuple[str, str, str]:
        stem = normalize_text(path.stem.replace("_", " "))
        # Law API snapshots are deliberately prefixed with LAW_.  Keep their
        # title and a stable per-law code so citations and LAW-only retrieval
        # remain distinct from KGS codes and internal rules.
        if re.match(r"(?i)^LAW\s+", stem):
            law_title = re.sub(r"(?i)^LAW\s+", "", stem)
            law_title = re.sub(r"\s+20\d{6}\s*$", "", law_title).strip(" -_")
            law_title = law_title or "법령 원문"
            return "LAW", f"LAW-{law_title}", law_title
        leading_rule_number = re.match(r"^\s*(\d{4,5}(?:-\d+)?)\b", stem)
        code_match = re.search(r"(?i)(?:^|\s)(?:KGS\s*)?(?:S\s*)?([A-Z]{2})\s*(\d{3})(?:\s|$)", stem)
        if not code_match and not leading_rule_number:
            code_match = re.search(r"(?i)KGS\s+(?:S\s*)?([A-Z]{2})\s*(\d{3})", first_pages[:5000])
        is_code = code_match is not None
        doc_type = "CODE" if is_code else "RULE"
        if code_match:
            doc_code = f"{code_match.group(1).upper()}{code_match.group(2)}"
        else:
            document_number = None
            for raw_line in first_pages.splitlines()[:40]:
                line = normalize_text(raw_line)
                exact_number = re.fullmatch(r"(\d{4,5}(?:-\d+)?)", line)
                if exact_number:
                    document_number = exact_number.group(1)
                    if "-" in document_number:
                        break
                if RULE_KIND_RE.search(line):
                    number = RULE_NUMBER_RE.search(line)
                    if number and not re.fullmatch(r"(?:19|20)\d{2}", number.group(1)):
                        document_number = number.group(1)
                        break
            doc_code = document_number or (leading_rule_number.group(1) if leading_rule_number else "UNKNOWN")

        title = stem
        title = re.sub(r"(?i)\.pdf$|\b전문\b|\b개정\b", " ", title)
        title = normalize_text(title)
        title = re.sub(r"^(?:KGS\s*)?[A-Z]{2}\s*\d{3}\s*[-–—]?\s*", "", title, flags=re.I)
        title = re.sub(r"^\d{4,5}(?:-\d+)?\s*[-–—]?\s*", "", title)
        title = re.sub(r"\b20\d{6}(?:\s+\d{6})?\b.*$", "", title).strip(" -_")
        title = re.sub(r"\([^)]*(?:19|20)\d{2}[^)]*\)", " ", title)
        if is_code:
            korean_lines = []
            for raw_line in first_pages.splitlines()[:15]:
                line = normalize_text(raw_line)
                if not line:
                    continue
                if "KGS " in line.upper() or re.search(r"20\d{2}[년.]", line) or re.search(r"[A-Za-z]{4,}", line):
                    if korean_lines:
                        break
                    continue
                no_space = line.replace(" ", "")
                if any(noise in no_space for noise in ("가스기술기준위원회", "위원장", "산업통상자원부", "심의·의결")):
                    if korean_lines:
                        break
                    continue
                if sum("가" <= char <= "힣" for char in line) >= 5:
                    korean_lines.append(line)
            if korean_lines:
                title = normalize_text(" ".join(korean_lines))
        else:
            for raw_line in first_pages.splitlines()[:40]:
                line = normalize_text(raw_line)
                if not RULE_KIND_RE.search(line) or re.search(r"(?:19|20)\d{2}\s*[.년]", line):
                    continue
                candidate = RULE_NUMBER_RE.sub(" ", line, count=1)
                candidate = normalize_text(candidate).strip(" -_[]【】")
                if len(candidate) >= 3:
                    title = candidate
                    break
        title = normalize_text(title).strip(" -_")
        return doc_type, doc_code, title or path.stem

    def _split_block(self, heading: str, block: str, page_number: int, doc_code: str) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        start = 0
        while start < len(block):
            end = min(len(block), start + self.chunk_chars)
            if end < len(block):
                boundaries = [
                    block.rfind(marker, start + self.chunk_chars // 2, end)
                    for marker in (". ", "다. ", "; ")
                ]
                boundary = max(boundaries)
                if boundary > start:
                    end = boundary + 1
            content = block[start:end].strip()
            if len(content) >= 30:
                chunks.append(
                    {
                        "hierarchy": heading,
                        "page": page_number,
                        "content": content,
                        "search_text": build_search_text(doc_code, heading, content),
                    }
                )
            if end >= len(block):
                break
            start = max(start + 1, end - self.overlap_chars)
        return chunks

    def _page_chunks(self, page_text: str, page_number: int, doc_code: str) -> list[dict[str, object]]:
        lines = [normalize_text(line) for line in page_text.splitlines()]
        lines = [line for line in lines if line and not re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line)]
        if not lines:
            return []

        hierarchy = f"[{doc_code}] {page_number}페이지"
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        current_heading = hierarchy
        for line in lines:
            if HEADING_RE.match(line) and len(line) <= 140:
                if buffer:
                    blocks.append((current_heading, normalize_text(" ".join(buffer))))
                    buffer = []
                current_heading = f"[{doc_code}] {line}"
                if re.match(r"^\d+(?:\.\d+){1,5}\.?\s+", line):
                    # Numbered code clauses may contain part or all of their rule text
                    # on the heading line. Retain it so wrapped and one-line clauses
                    # are both fully searchable and available to grounded answers.
                    buffer.append(line)
            else:
                buffer.append(line)
        if buffer:
            blocks.append((current_heading, normalize_text(" ".join(buffer))))

        chunks: list[dict[str, object]] = []
        for heading, block in blocks:
            chunks.extend(self._split_block(heading, block, page_number, doc_code))
        return chunks

    def _rule_chunks(self, page_texts: list[str], doc_code: str) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        chapter = ""
        heading = f"[{doc_code}] 문서 서두"
        buffer: list[str] = []
        start_page = 1

        def flush() -> None:
            nonlocal buffer
            block = normalize_text(" ".join(buffer))
            if block:
                chunks.extend(self._split_block(heading, block, start_page, doc_code))
            buffer = []

        for page_number, page_text in enumerate(page_texts, 1):
            for raw_line in page_text.splitlines():
                line = normalize_text(raw_line)
                if not line or re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line):
                    continue
                chapter_match = RULE_CHAPTER_RE.match(line)
                if chapter_match:
                    flush()
                    chapter = normalize_text(f"{chapter_match.group('number')} {chapter_match.group('title')}")
                    heading = f"[{doc_code}] {chapter}"
                    start_page = page_number
                    continue
                article_match = RULE_ARTICLE_RE.match(line)
                if article_match:
                    flush()
                    article = normalize_text(
                        f"{article_match.group('number')}{article_match.group('title') or ''}"
                    )
                    heading = f"[{doc_code}] " + (f"{chapter} > " if chapter else "") + article
                    start_page = page_number
                    body = normalize_text(article_match.group("body"))
                    if body:
                        buffer.append(body)
                    continue
                if line.startswith("부칙") and len(line) <= 80:
                    flush()
                    heading = f"[{doc_code}] {line}"
                    start_page = page_number
                    continue
                buffer.append(line)
        flush()
        return chunks

    def chunks_from_segments(self, doc_code: str, segments: Iterable[tuple[str, int, str]]) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        for hierarchy, page, content in segments:
            normalized_hierarchy = normalize_text(hierarchy)
            normalized_hierarchy = re.sub(r"^\[[^]]+\]", f"[{doc_code}]", normalized_hierarchy)
            chunks.extend(self._split_block(normalized_hierarchy, normalize_text(content), page, doc_code))
        return chunks

    def index_pdf(self, path: Path, force: bool = False) -> IndexResult:
        path = path.resolve()
        file_hash = self.hash_file(path)
        existing = self.database.document_by_filename(path.name)
        if existing and existing["file_hash"] == file_hash and not force:
            return IndexResult(path.name, "unchanged", int(existing["id"]))
        try:
            with pymupdf.open(path) as document:
                page_texts: list[str] = []
                needs_ocr = False
                for page in document:
                    text = page.get_text("text")
                    if text.count("\ufffd") >= 3:
                        needs_ocr = True
                        text = self.ocr_page(page) if self.ocr_page else ""
                    page_texts.append(text)
                preview = "\n".join(page_texts[:3])
                doc_type, doc_code, title = self.infer_metadata(path, preview)
                chunks: list[dict[str, object]] = []
                if doc_type in {"RULE", "LAW"}:
                    chunks = self._rule_chunks(page_texts, doc_code)
                else:
                    for page_index, page_text in enumerate(page_texts, start=1):
                        chunks.extend(self._page_chunks(page_text, page_index, doc_code))
                metadata = {
                    "doc_type": doc_type,
                    "doc_code": doc_code,
                    "title": title,
                    "filename": path.name,
                    "file_path": str(path),
                    "file_hash": file_hash,
                    "page_count": len(document),
                }
            document_id = self.database.replace_document(metadata, chunks)
            status = "needs_ocr" if needs_ocr and not self.ocr_page else "indexed"
            return IndexResult(path.name, status, document_id, len(chunks))
        except Exception as exc:
            return IndexResult(path.name, "failed", error=str(exc))

    def index_directory(self, directory: Path, force: bool = False, progress: Callable[[int, int, IndexResult], None] | None = None) -> list[IndexResult]:
        paths: Iterable[Path] = sorted(directory.glob("*.pdf"), key=lambda item: item.name.lower())
        paths = list(paths)
        results: list[IndexResult] = []
        for index, path in enumerate(paths, start=1):
            result = self.index_pdf(path, force=force)
            results.append(result)
            if progress:
                progress(index, len(paths), result)
        return results
