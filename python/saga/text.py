from __future__ import annotations

import re
import unicodedata
import json
from functools import lru_cache
from pathlib import Path


WORD_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}\d*|\d{2,}")
CODE_RE = re.compile(r"(?i)(?:KGS[\s_-]*)?([A-Z]{2})[\s_-]?(\d{3})")
STOPWORDS = {
    "알려줘", "알려주세요", "설명", "설명해줘", "무엇", "뭐야", "어떻게", "관련", "대한",
    "기준", "규정", "지침", "사규", "내용", "문서", "주요", "핵심", "적용", "범위",
    "kgs", "code", "대해서", "있는지", "인가요", "해주세요", "정리", "절차",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "").replace("\u200b", "")
    return re.sub(r"\s+", " ", value).strip()


def extract_code(value: str) -> str:
    match = CODE_RE.search(value or "")
    if match:
        return f"{match.group(1).upper()}{match.group(2)}"
    number = re.search(r"(?<!\d)(\d{4,5}(?:-\d+)?)(?!\d)", value or "")
    return number.group(1) if number else "UNKNOWN"


def extract_codes(value: str) -> list[str]:
    """Return every explicit KGS code in appearance order, without duplicates."""
    codes: list[str] = []
    for match in CODE_RE.finditer(value or ""):
        code = f"{match.group(1).upper()}{match.group(2)}"
        if code not in codes:
            codes.append(code)
    return codes


def extract_document_codes(value: str) -> list[str]:
    """Extract explicit KGS codes and clearly identified internal-rule numbers."""
    codes = extract_codes(value)
    normalized = normalize_text(value)
    has_rule_context = bool(re.search(r"(규정|지침|요령|정관|규칙|강령|문서(?:번호)?)", normalized))
    for match in re.finditer(r"(?<!\d)(\d{4,5}(?:-\d+)?)(?!\d)", normalized):
        number = match.group(1)
        trailing = normalized[match.end():].lstrip()
        if re.fullmatch(r"(?:19|20)\d{2}", number) and trailing.startswith("년"):
            continue
        if len(number) == 5 or "-" in number or has_rule_context:
            if number not in codes:
                codes.append(number)
    return codes


def search_tokens(value: str, max_tokens: int = 180) -> list[str]:
    """Create language-independent word and n-gram tokens for Korean technical text."""
    normalized = normalize_text(value).lower()
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip("_-.")
        if len(token) < 2 or token in seen:
            return
        seen.add(token)
        tokens.append(token)

    for match in WORD_RE.finditer(normalized):
        word = match.group(0)
        add(word)
        if re.fullmatch(r"[가-힣]{4,}", word):
            for size in (2, 3):
                for index in range(len(word) - size + 1):
                    add(word[index : index + size])
        if len(tokens) >= max_tokens:
            break
    return tokens[:max_tokens]


def build_search_text(*values: str) -> str:
    return " ".join(search_tokens(" ".join(values)))


def meaningful_terms(value: str, max_terms: int = 16) -> list[str]:
    """Extract user concepts for result coverage scoring without n-gram inflation."""
    terms: list[str] = []
    for match in WORD_RE.finditer(normalize_text(value).lower()):
        term = match.group(0).strip("_-.")
        if term in STOPWORDS or term.isdigit() or term in terms:
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


@lru_cache(maxsize=1)
def synonym_groups() -> tuple[tuple[str, ...], ...]:
    root = Path(__file__).resolve().parents[2]
    path = root / "synonyms.txt"
    groups: list[tuple[str, ...]] = []
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = tuple(
                dict.fromkeys(normalize_text(value).lower() for value in line.split(",") if normalize_text(value))
            )
            if len(values) >= 2:
                groups.append(values)
    # The structured dictionary carries relation/ambiguity metadata.  Only
    # high-confidence entries marked ``expand`` participate in lexical query
    # expansion; related or ambiguous terms remain available for audit and can
    # be handled by the clarification router instead of being conflated.
    structured_path = root / "term_aliases.json"
    if structured_path.exists():
        try:
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
            for entry in payload.get("entries", []):
                if not entry.get("expand"):
                    continue
                values = tuple(
                    dict.fromkeys(
                        normalize_text(value).lower()
                        for value in [entry.get("canonical", ""), *entry.get("aliases", [])]
                        if normalize_text(value)
                    )
                )
                if len(values) >= 2:
                    groups.append(values)
        except (OSError, ValueError, TypeError):
            # A malformed optional dictionary must not disable the original
            # synonyms.txt fallback or prevent the server from starting.
            pass
    return tuple(groups)


def expand_with_synonyms(value: str, max_additions: int = 16) -> str:
    normalized = normalize_text(value)
    lower = normalized.lower()
    additions: list[str] = []
    for group in synonym_groups():
        if not any(alias in lower for alias in group):
            continue
        for alias in group:
            if alias not in lower and alias not in additions:
                additions.append(alias)
                if len(additions) >= max_additions:
                    return normalize_text(" ".join([normalized, *additions]))
    return normalize_text(" ".join([normalized, *additions]))


def fts_query(value: str) -> str:
    tokens = search_tokens(value, max_tokens=40)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
