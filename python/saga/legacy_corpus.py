"""Recover high-quality document metadata and section boundaries from the legacy index export."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .text import normalize_text


LEGACY_CODE_RE = re.compile(r"(?i)(?:KGS\s*)?(?:S\s*)?([A-Z]{2})\s*(\d{3})")
RULE_NUMBER_RE = re.compile(r"(?<!\d)(\d{4,5}(?:-\d+)?)(?!\d)")
DATE_NOISE_RE = re.compile(
    r"\([^)]*(?:19|20)\d{2}[^)]*\)|"
    r"\b(?:19|20)\d{6}\b|\b\d{6}\b|"
    r"(?:19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일"
)


@dataclass(slots=True)
class LegacySegment:
    hierarchy: str
    page: int
    content: str


@dataclass(slots=True)
class LegacyDocument:
    doc_type: str
    raw_code: str
    doc_code: str
    title: str
    segments: list[LegacySegment] = field(default_factory=list)


@dataclass(slots=True)
class SourcePdf:
    path: Path
    file_hash: str
    filename_code: str | None
    filename_title: str


@dataclass(slots=True)
class CorpusMatch:
    source: SourcePdf
    legacy: LegacyDocument
    score: float


def _decode(value: str) -> str:
    return base64.b64decode(value).decode("utf-8", errors="replace")


def normalize_legacy_code(doc_type: str, raw_code: str) -> str:
    if doc_type.upper() == "CODE":
        match = LEGACY_CODE_RE.search(raw_code)
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
    match = RULE_NUMBER_RE.search(raw_code)
    return match.group(1) if match else "UNKNOWN"


def clean_legacy_title(value: str) -> str:
    value = normalize_text(value).replace("[TITLE]", "")
    value = re.sub(r"\s+([·․ㆍᆞ])\s+", r"\1", value)
    value = re.sub(r"(?<=[가-힣])\s+(?=[가-힣](?:\s|$))", "", value)
    return normalize_text(value).strip(" -_·")


def title_key(value: str) -> str:
    value = normalize_text(value).lower().replace("[title]", "")
    value = DATE_NOISE_RE.sub(" ", value)
    value = re.sub(r"(?i)\.pdf$|전문|개정안?|제정전문|개정전문", " ", value)
    value = re.sub(r"\b\d{4,5}(?:-\d+)?\b", " ", value)
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def filename_metadata(path: Path) -> tuple[str | None, str]:
    stem = normalize_text(path.stem.replace("_", " "))
    code_match = LEGACY_CODE_RE.search(stem)
    if code_match:
        code = f"{code_match.group(1).upper()}{code_match.group(2)}"
    else:
        leading_number = re.match(r"^\s*(\d{4,5}(?:-\d+)?)\b", stem)
        if leading_number:
            code = leading_number.group(1)
        else:
            candidates = RULE_NUMBER_RE.findall(stem)
            code = next((item for item in candidates if not re.fullmatch(r"(?:19|20)\d{2}", item)), None)
    return code, title_key(stem)


def load_legacy_export(path: Path) -> list[LegacyDocument]:
    grouped: dict[tuple[str, str, str], LegacyDocument] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 7:
                continue
            decoded = [_decode(value) for value in columns[:7]]
            doc_type, raw_code, raw_title, _summary, hierarchy, page_value, content = decoded
            title = clean_legacy_title(raw_title)
            key = (doc_type, raw_code, title)
            document = grouped.setdefault(
                key,
                LegacyDocument(doc_type, raw_code, normalize_legacy_code(doc_type, raw_code), title),
            )
            try:
                page = max(1, int(page_value))
            except ValueError:
                page = 1
            content = normalize_text(content)
            if content:
                document.segments.append(LegacySegment(normalize_text(hierarchy), page, content))
    return list(grouped.values())


def source_pdfs(paths: list[Path]) -> tuple[list[SourcePdf], list[list[str]]]:
    by_hash: dict[str, list[Path]] = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path)
    sources: list[SourcePdf] = []
    duplicates: list[list[str]] = []
    for digest, group in by_hash.items():
        group.sort(key=lambda item: item.name.lower())
        canonical = group[0]
        code, title = filename_metadata(canonical)
        sources.append(SourcePdf(canonical, digest, code, title))
        if len(group) > 1:
            duplicates.append([item.name for item in group])
    return sources, duplicates


def _title_similarity(source: SourcePdf, legacy: LegacyDocument) -> float:
    left, right = source.filename_title, title_key(legacy.title)
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(None, left, right).ratio()
    if left in right or right in left:
        ratio = max(ratio, min(len(left), len(right)) / max(len(left), len(right)) + 0.15)
    return min(ratio, 1.0)


def match_corpus(sources: list[SourcePdf], legacy_documents: list[LegacyDocument]) -> tuple[list[CorpusMatch], list[SourcePdf], list[LegacyDocument]]:
    proposals: list[tuple[float, int, int]] = []
    for legacy_index, legacy in enumerate(legacy_documents):
        for source_index, source in enumerate(sources):
            title_score = _title_similarity(source, legacy)
            code_match = source.filename_code == legacy.doc_code
            legacy_title_key = title_key(legacy.title)
            title_contains = bool(source.filename_title and legacy_title_key) and (
                source.filename_title in legacy_title_key or legacy_title_key in source.filename_title
            )
            if legacy.doc_type == "CODE" and code_match:
                score = 200.0 + 20.0 * title_score
            elif title_score >= 0.85:
                score = 250.0 + 20.0 * title_score + (10.0 if code_match else 0.0)
            elif code_match and title_contains:
                score = 180.0 + 20.0 * title_score
            elif code_match and title_score >= 0.55:
                score = 130.0 + 50.0 * title_score
            elif title_score >= 0.62:
                score = 100.0 * title_score
            else:
                continue
            proposals.append((score, source_index, legacy_index))
    proposals.sort(reverse=True)
    used_sources: set[int] = set()
    used_legacy: set[int] = set()
    matches: list[CorpusMatch] = []
    for score, source_index, legacy_index in proposals:
        if source_index in used_sources or legacy_index in used_legacy:
            continue
        used_sources.add(source_index)
        used_legacy.add(legacy_index)
        matches.append(CorpusMatch(sources[source_index], legacy_documents[legacy_index], score))
    matches.sort(key=lambda item: item.source.path.name.lower())
    unmatched_sources = [item for index, item in enumerate(sources) if index not in used_sources]
    unmatched_legacy = [item for index, item in enumerate(legacy_documents) if index not in used_legacy]
    return matches, unmatched_sources, unmatched_legacy
