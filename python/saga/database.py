from __future__ import annotations

import json
import heapq
import re
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

from .text import (
    build_search_text,
    extract_codes,
    extract_document_codes,
    fts_query,
    meaningful_terms,
    normalize_text,
)
from .vector_index import VECTOR_DIM, VECTOR_MODEL, cosine, decode, encode, query_vector


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type TEXT NOT NULL,
    doc_code TEXT NOT NULL,
    title TEXT NOT NULL,
    filename TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    hierarchy TEXT NOT NULL DEFAULT '',
    parent_hierarchy TEXT NOT NULL DEFAULT '',
    section_number TEXT NOT NULL DEFAULT '',
    content_kind TEXT NOT NULL DEFAULT 'paragraph',
    table_json TEXT NOT NULL DEFAULT '',
    page INTEGER NOT NULL DEFAULT 1,
    content TEXT NOT NULL,
    search_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    search_text,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_v2 USING fts5(
    chunk_id UNINDEXED,
    code_terms,
    title_terms,
    hierarchy_terms,
    content_terms,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL,
    intent TEXT NOT NULL,
    references_json TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    feedback_score INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hazop_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    no INTEGER,
    scenario_id TEXT NOT NULL DEFAULT '*',
    scenario_name TEXT NOT NULL DEFAULT '',
    tag_id TEXT NOT NULL,
    item_name TEXT NOT NULL DEFAULT '',
    unit TEXT NOT NULL DEFAULT '',
    normal_range TEXT NOT NULL DEFAULT '',
    relevance TEXT NOT NULL DEFAULT '직접관련',
    guide_word TEXT NOT NULL DEFAULT '',
    cond_text TEXT NOT NULL DEFAULT '',
    threshold_type TEXT NOT NULL DEFAULT 'STATIC',
    threshold_value REAL,
    threshold_basis TEXT NOT NULL DEFAULT '',
    threshold_source TEXT NOT NULL DEFAULT '',
    threshold_confidence TEXT NOT NULL DEFAULT 'unverified',
    compare_dir TEXT NOT NULL DEFAULT '',
    severity TEXT NOT NULL DEFAULT '주의',
    severity_rank INTEGER NOT NULL DEFAULT 1,
    risk_scenario TEXT NOT NULL DEFAULT '',
    consequence TEXT NOT NULL DEFAULT '',
    emergency_action TEXT NOT NULL DEFAULT '',
    future_measure TEXT NOT NULL DEFAULT '',
    standard_ref TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'digital-twin-hazop',
    source_url TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(scenario_id, tag_id, no, cond_text)
);

CREATE TABLE IF NOT EXISTS hazop_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_code ON documents(doc_code);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_hazop_rules_scenario_tag ON hazop_rules(scenario_id, tag_id);
CREATE INDEX IF NOT EXISTS idx_hazop_evaluations_station_time ON hazop_evaluations(station_id, id DESC);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._write_lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _has_document_column(self, name: str) -> bool:
        """Keep read paths compatible with databases created before migrations."""
        with self.connect() as connection:
            return any(
                str(row[1]) == name
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            )

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_schema(connection)
        self._sync_search_index()
        self._sync_chunk_metadata()
        self._sync_vector_index()

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Add structured chunk columns to databases created by v2.0."""
        existing = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        existing_documents = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        existing_hazop = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(hazop_rules)").fetchall()
        }
        migrations = {
            "parent_hierarchy": "ALTER TABLE chunks ADD COLUMN parent_hierarchy TEXT NOT NULL DEFAULT ''",
            "section_number": "ALTER TABLE chunks ADD COLUMN section_number TEXT NOT NULL DEFAULT ''",
            "content_kind": "ALTER TABLE chunks ADD COLUMN content_kind TEXT NOT NULL DEFAULT 'paragraph'",
            "table_json": "ALTER TABLE chunks ADD COLUMN table_json TEXT NOT NULL DEFAULT ''",
            "source_url": "ALTER TABLE documents ADD COLUMN source_url TEXT NOT NULL DEFAULT ''",
            "threshold_basis": "ALTER TABLE hazop_rules ADD COLUMN threshold_basis TEXT NOT NULL DEFAULT ''",
            "threshold_source": "ALTER TABLE hazop_rules ADD COLUMN threshold_source TEXT NOT NULL DEFAULT ''",
            "threshold_confidence": "ALTER TABLE hazop_rules ADD COLUMN threshold_confidence TEXT NOT NULL DEFAULT 'unverified'",
        }
        for column, statement in migrations.items():
            available = (
                existing_documents if column == "source_url"
                else existing_hazop if column.startswith("threshold_")
                else existing
            )
            if column not in available:
                connection.execute(statement)

    def _sync_chunk_metadata(self) -> None:
        """Backfill structured metadata for chunks created before v2.1."""
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id, hierarchy, content, section_number, content_kind, table_json FROM chunks"
            ).fetchall()
            updates: list[tuple[str, str, str, str, int]] = []
            for row in rows:
                hierarchy = normalize_text(str(row["hierarchy"]))
                content = normalize_text(str(row["content"]))
                section_match = re.search(
                    r"(?<!\d)(\d+(?:\.\d+){0,5})(?:\s|$)", hierarchy
                )
                section = section_match.group(1) if section_match else ""
                parent = hierarchy.rsplit(" > ", 1)[0] if " > " in hierarchy else ""
                kind = "table" if self._looks_like_table(content) else "paragraph"
                table_json = self._table_metadata(content) if kind == "table" else ""
                if (
                    str(row["section_number"] or "") == section
                    and str(row["content_kind"] or "paragraph") == kind
                    and (kind != "table" or bool(str(row["table_json"] or "")))
                ):
                    continue
                updates.append((parent, section, kind, table_json, int(row["id"])))
            if updates:
                connection.executemany(
                    "UPDATE chunks SET parent_hierarchy = ?, section_number = ?, content_kind = ?, table_json = ? WHERE id = ?",
                    updates,
                )

    @staticmethod
    def _looks_like_table(content: str) -> bool:
        lines = [line.strip() for line in re.split(r"[\n|]", content) if line.strip()]
        if "|" in content:
            return True
        if re.search(r"(?:표|table)\s*\d", content, re.IGNORECASE) and len(
            re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:m3|m³|m|kg|kPa|MPa|%)?", content, re.IGNORECASE)
        ) >= 4:
            return True
        # PDF text extraction often flattens columns into repeated numeric
        # cells.  Mark only strongly table-like content to avoid polluting prose.
        return len(lines) >= 4 and sum(bool(re.search(r"\d", line)) for line in lines) >= 3

    @staticmethod
    def _table_metadata(content: str) -> str:
        values = re.findall(
            r"\d[\d,]*(?:\.\d+)?\s*(?:m3|m³|m|kg|kPa|MPa|%)?",
            content,
            re.IGNORECASE,
        )
        return json.dumps({"numeric_values": values[:128]}, ensure_ascii=False)

    def _sync_vector_index(self) -> None:
        """Build the dependency-free fallback vector index incrementally."""
        with self.connect() as connection:
            missing = connection.execute(
                """SELECT c.id, c.search_text
                     FROM chunks c
                     LEFT JOIN chunk_vectors v ON v.chunk_id = c.id AND v.model = ?
                    WHERE v.chunk_id IS NULL
                    ORDER BY c.id""",
                (VECTOR_MODEL,),
            ).fetchall()
        if not missing:
            return
        now = datetime.now(UTC).isoformat()
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunk_vectors (chunk_id, model, dimension, vector, updated_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (int(row["id"]), VECTOR_MODEL, VECTOR_DIM, encode(str(row["search_text"])), now)
                    for row in missing
                ],
            )

    def _sync_search_index(self) -> None:
        """Backfill the weighted FTS index after upgrading an existing database."""
        with self.connect() as connection:
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            index_count = int(connection.execute("SELECT COUNT(*) FROM chunks_fts_v2").fetchone()[0])
        if chunk_count == index_count:
            return
        with self.transaction() as connection:
            connection.execute("DELETE FROM chunks_fts_v2")
            rows = connection.execute(
                """SELECT c.id, c.hierarchy, c.content, d.doc_code, d.title
                     FROM chunks c JOIN documents d ON d.id = c.document_id
                    ORDER BY c.id"""
            )
            batch: list[tuple[int, str, str, str, str]] = []
            for row in rows:
                batch.append(
                    (
                        int(row["id"]),
                        build_search_text(str(row["doc_code"])),
                        build_search_text(str(row["title"])),
                        build_search_text(str(row["hierarchy"])),
                        build_search_text(str(row["content"])),
                    )
                )
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO chunks_fts_v2 VALUES (?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO chunks_fts_v2 VALUES (?, ?, ?, ?, ?)", batch)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def document_by_filename(self, filename: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM documents WHERE filename = ?", (filename,)
            ).fetchone()

    def replace_document(self, metadata: dict[str, Any], chunks: list[dict[str, Any]]) -> int:
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE filename = ?", (metadata["filename"],)
            ).fetchone()
            if existing:
                document_id = int(existing["id"])
                old_ids = [row["id"] for row in connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
                )]
                if old_ids:
                    marks = ",".join("?" for _ in old_ids)
                    connection.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", old_ids)
                    connection.execute(f"DELETE FROM chunks_fts_v2 WHERE chunk_id IN ({marks})", old_ids)
                    connection.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({marks})", old_ids)
                    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute(
                    """UPDATE documents
                          SET doc_type = ?, doc_code = ?, title = ?, file_path = ?,
                              file_hash = ?, page_count = ?, indexed_at = ?, source_url = ?
                        WHERE id = ?""",
                    (
                        metadata["doc_type"], metadata["doc_code"], metadata["title"],
                        metadata["file_path"], metadata["file_hash"], metadata["page_count"],
                        datetime.now(UTC).isoformat(), metadata.get("source_url", ""), document_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """INSERT INTO documents
                       (doc_type, doc_code, title, filename, file_path, file_hash, page_count, indexed_at, source_url)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        metadata["doc_type"], metadata["doc_code"], metadata["title"],
                        metadata["filename"], metadata["file_path"], metadata["file_hash"],
                        metadata["page_count"], datetime.now(UTC).isoformat(), metadata.get("source_url", ""),
                    ),
                )
                document_id = int(cursor.lastrowid)

            for chunk in chunks:
                hierarchy = str(chunk.get("hierarchy", ""))
                content = str(chunk.get("content", ""))
                section_match = re.search(r"(?<!\d)(\d+(?:\.\d+){0,5})(?:\s|$)", hierarchy)
                section_number = str(chunk.get("section_number") or (section_match.group(1) if section_match else ""))
                parent_hierarchy = str(
                    chunk.get("parent_hierarchy")
                    or (hierarchy.rsplit(" > ", 1)[0] if " > " in hierarchy else "")
                )
                content_kind = str(chunk.get("content_kind") or ("table" if self._looks_like_table(content) else "paragraph"))
                table_json = str(chunk.get("table_json") or (self._table_metadata(content) if content_kind == "table" else ""))
                chunk_cursor = connection.execute(
                    """INSERT INTO chunks
                       (document_id, hierarchy, parent_hierarchy, section_number, content_kind, table_json, page, content, search_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (document_id, hierarchy, parent_hierarchy, section_number, content_kind, table_json,
                     chunk["page"], content, chunk["search_text"]),
                )
                chunk_id = int(chunk_cursor.lastrowid)
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, search_text) VALUES (?, ?)",
                    (chunk_id, chunk["search_text"]),
                )
                connection.execute(
                    "INSERT INTO chunks_fts_v2 VALUES (?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        build_search_text(metadata["doc_code"]),
                        build_search_text(metadata["title"]),
                        build_search_text(chunk["hierarchy"]),
                        build_search_text(chunk["content"]),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_vectors (chunk_id, model, dimension, vector, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, VECTOR_MODEL, VECTOR_DIM, encode(str(chunk["search_text"])), datetime.now(UTC).isoformat()),
                )
            return document_id

    def update_document_metadata(self, filename: str, **updates: str) -> bool:
        """Update provenance metadata for a downloaded non-PDF source."""
        allowed = {"doc_type", "doc_code", "title", "file_path", "file_hash", "source_url"}
        values = {key: value for key, value in updates.items() if key in allowed and value is not None}
        if not values:
            return False
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {assignments}, indexed_at = ? WHERE filename = ?",
                [*values.values(), datetime.now(UTC).isoformat(), filename],
            )
            return cursor.rowcount > 0

    def replace_source_records(
        self,
        *,
        doc_type: str,
        doc_code: str,
        title: str,
        filename: str,
        file_path: str,
        file_hash: str,
        source_url: str,
        records: list[dict[str, Any]],
    ) -> int:
        """Store structured external records in the same searchable RAG index.

        Each row becomes a bounded, self-contained chunk. This keeps the
        operational/incident corpus queryable by both FTS and the existing
        local vector fallback while retaining the original spreadsheet as the
        downloadable source artifact.
        """
        chunks: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            labels = []
            for key, value in record.items():
                text = normalize_text(str(value))
                if text and text.lower() not in {"nan", "none", "null"}:
                    labels.append(f"{key}: {text}")
            if not labels:
                continue
            content = "\n".join(labels)
            hierarchy = f"[{doc_code}] 레코드 {index}"
            chunks.append({
                "hierarchy": hierarchy,
                "page": index,
                "content": content,
                "search_text": build_search_text(doc_code, title, hierarchy, content),
            })
        return self.replace_document(
            {
                "doc_type": doc_type,
                "doc_code": doc_code,
                "title": title,
                "filename": filename,
                "file_path": file_path,
                "file_hash": file_hash,
                "page_count": len(chunks),
                "source_url": source_url,
            },
            chunks,
        )

    def vector_search(
        self,
        query: str,
        limit: int,
        doc_codes: list[str] | None = None,
        domain: str = "CROSS",
        source_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the local hashed-vector index with metadata constraints."""
        query_values = query_vector(query)
        conditions = ["v.model = ?", "v.dimension = ?"]
        params: list[Any] = [VECTOR_MODEL, VECTOR_DIM]
        if doc_codes:
            conditions.append(f"d.doc_code IN ({','.join('?' for _ in doc_codes)})")
            params.extend(doc_codes)
        if source_types:
            conditions.append(f"d.doc_type IN ({','.join('?' for _ in source_types)})")
            params.extend(source_types)
        if domain == "CODE":
            conditions.append("d.doc_type = 'CODE'")
        elif domain == "RULE":
            conditions.append("d.doc_type = 'RULE'")
        elif domain == "LAW":
            conditions.append("d.doc_type = 'LAW'")
        source_url = "d.source_url" if self._has_document_column("source_url") else "''"
        sql = f"""
            SELECT c.id AS chunk_id, c.document_id, c.hierarchy, c.parent_hierarchy,
                   c.section_number, c.content_kind, c.table_json, c.page, c.content,
                   d.doc_type, d.doc_code, d.title, d.filename, d.file_path, {source_url} AS source_url, v.vector
              FROM chunk_vectors v
              JOIN chunks c ON c.id = v.chunk_id
              JOIN documents d ON d.id = c.document_id
             WHERE {' AND '.join(conditions)}
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            item.pop("vector", None)
            score = cosine(query_values, decode(row["vector"]))
            item["score"] = round(score, 6)
            item["retrieval_method"] = "vector"
            ranked.append((score, int(row["chunk_id"]), item))
        top = heapq.nlargest(max(limit * 5, 80), ranked, key=lambda value: (value[0], -value[1]))
        return [item for _, _, item in top[:limit]]

    def hybrid_search(
        self,
        query: str,
        limit: int,
        doc_codes: list[str] | None = None,
        domain: str = "CROSS",
        source_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse lexical FTS and local vector retrieval with reciprocal rank fusion."""
        # An explicit KGS/rule identifier is a hard document constraint.  Do
        # this at the database boundary as well as in the RAG planner so
        # direct retrieval tests and lightweight API users cannot let a
        # semantically similar but unrelated code outrank the requested one.
        if not doc_codes and domain == "CODE":
            explicit_codes = extract_codes(query)
            if explicit_codes:
                doc_codes = explicit_codes
        elif not doc_codes and domain == "RULE":
            explicit_codes = extract_document_codes(query)
            if explicit_codes:
                doc_codes = explicit_codes
        lexical = self.search(query, max(limit * 2, limit), doc_codes, domain, source_types)
        vector = self.vector_search(query, max(limit * 2, limit), doc_codes, domain, source_types)
        by_id: dict[int, dict[str, Any]] = {}
        fused: dict[int, float] = defaultdict(float)
        # Keep exact regulatory terms slightly ahead of character-n-gram
        # similarity. The vector side is a recall backstop, not a licence to
        # outrank an exact clause/number hit.
        for rank, item in enumerate(lexical, start=1):
            chunk_id = int(item["chunk_id"])
            by_id[chunk_id] = item
            fused[chunk_id] += 1.5 / (60.0 + rank)
            item["retrieval_method"] = "lexical"
        for rank, item in enumerate(vector, start=1):
            chunk_id = int(item["chunk_id"])
            by_id.setdefault(chunk_id, item)
            fused[chunk_id] += 1.0 / (60.0 + rank)
            if chunk_id in by_id and by_id[chunk_id] is not item:
                by_id[chunk_id]["vector_score"] = item.get("score", 0.0)
            else:
                item["retrieval_method"] = "vector"
        # Always retain a small lexical anchor set. This prevents a vector
        # similarity tie from hiding the exact clause that contains a number,
        # table label or legal term (the usual failure mode for safety rules).
        lexical_anchor_ids = {
            int(item["chunk_id"])
            for item in lexical[: max(2, min(6, limit // 2))]
        }
        lexical_ranks = {
            int(item["chunk_id"]): rank
            for rank, item in enumerate(lexical, start=1)
        }
        ordered = sorted(
            by_id.values(),
            key=lambda item: (
                0 if int(item["chunk_id"]) in lexical_anchor_ids else 1,
                lexical_ranks.get(int(item["chunk_id"]), 10_000)
                if int(item["chunk_id"]) in lexical_anchor_ids
                else -fused[int(item["chunk_id"])],
                int(item["document_id"]),
                int(item["page"]),
                int(item["chunk_id"]),
            ),
        )
        selected: list[dict[str, Any]] = []
        per_document: dict[int, int] = defaultdict(int)
        # A source-filtered external corpus often contains one document with
        # many meaningful records (for example the NREL report or HIAD
        # workbook).  Capping it at four chunks would discard the very rows
        # that matched the operator's topic.  Keep the diversity cap for the
        # mixed standards corpus, but let an explicitly isolated source fill
        # the requested context window.
        per_document_limit = (
            limit
            if source_types and len(source_types) == 1
            else max(4, min(10, limit // 3))
        )
        for item in ordered:
            document_id = int(item["document_id"])
            if per_document[document_id] >= per_document_limit:
                continue
            item["score"] = round(fused[int(item["chunk_id"])], 6)
            item["retrieval_method"] = "hybrid"
            per_document[document_id] += 1
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def search(
        self,
        query: str,
        limit: int,
        doc_codes: list[str] | None = None,
        domain: str = "CROSS",
        source_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        expression = fts_query(query)
        if not expression:
            return []
        conditions = ["chunks_fts_v2 MATCH ?"]
        params: list[Any] = [expression]
        if doc_codes:
            conditions.append(f"d.doc_code IN ({','.join('?' for _ in doc_codes)})")
            params.extend(doc_codes)
        if source_types:
            conditions.append(f"d.doc_type IN ({','.join('?' for _ in source_types)})")
            params.extend(source_types)
        if domain == "CODE":
            conditions.append("d.doc_type = 'CODE'")
        elif domain == "RULE":
            conditions.append("d.doc_type = 'RULE'")
        elif domain == "LAW":
            conditions.append("d.doc_type = 'LAW'")
        candidate_limit = max(limit * 5, 80)
        params.append(candidate_limit)
        source_url = "d.source_url" if self._has_document_column("source_url") else "''"
        sql = f"""
            SELECT c.id AS chunk_id, c.document_id, c.hierarchy, c.page, c.content,
                   d.doc_type, d.doc_code, d.title, d.filename, d.file_path, {source_url} AS source_url,
                   bm25(chunks_fts_v2, 0.0, 9.0, 5.0, 3.0, 1.0) AS rank
              FROM chunks_fts_v2
              JOIN chunks c ON c.id = chunks_fts_v2.chunk_id
              JOIN documents d ON d.id = c.document_id
             WHERE {' AND '.join(conditions)}
             ORDER BY rank ASC
             LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        terms = meaningful_terms(query)
        normalized_query = normalize_text(query).lower()
        rescored: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            title = normalize_text(str(row["title"])).lower()
            hierarchy = normalize_text(str(row["hierarchy"])).lower()
            content = normalize_text(str(row["content"])).lower()
            combined = f"{title} {hierarchy} {content}"
            coverage = sum(term in combined for term in terms) / max(1, len(terms))
            heading_coverage = sum(term in hierarchy for term in terms) / max(1, len(terms))
            title_coverage = sum(term in title for term in terms) / max(1, len(terms))
            phrase_bonus = 1.2 if len(normalized_query) >= 4 and normalized_query in combined else 0.0
            lexical = max(0.0, -float(row["rank"]))
            item["score"] = round(
                lexical + coverage * 5.0 + heading_coverage * 2.5 + title_coverage * 2.0 + phrase_bonus,
                6,
            )
            rescored.append(item)
        rescored.sort(key=lambda item: (-item["score"], item["document_id"], item["page"], item["chunk_id"]))
        selected: list[dict[str, Any]] = []
        per_document: dict[int, int] = defaultdict(int)
        seen_content: set[str] = set()
        max_per_document = max(4, min(10, limit // 3))
        for item in rescored:
            content_key = normalize_text(item["content"])[:500]
            if content_key in seen_content or per_document[item["document_id"]] >= max_per_document:
                continue
            seen_content.add(content_key)
            per_document[item["document_id"]] += 1
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def search_headings(self, heading: str, doc_codes: list[str], limit: int = 8) -> list[dict[str, Any]]:
        if not doc_codes:
            return []
        marks = ",".join("?" for _ in doc_codes)
        sql = f"""
            SELECT c.id AS chunk_id, c.document_id, c.hierarchy, c.page, c.content,
                   d.doc_type, d.doc_code, d.title, d.filename, d.file_path
              FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE d.doc_code IN ({marks}) AND c.hierarchy LIKE ?
             ORDER BY c.page ASC, c.id ASC LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(sql, [*doc_codes, f"%{heading}%", limit]).fetchall()
        return [{**dict(row), "score": 100.0 - index} for index, row in enumerate(rows)]

    def search_pages(
        self,
        pages: list[int],
        doc_codes: list[str],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return indexed chunks on specific PDF pages for table continuations.

        PDF extraction does not always repeat a section heading on the next
        page, so heading-only retrieval cannot reliably recover multi-page
        tables.  This small page-scoped lookup keeps that continuation
        retrieval explicit and bounded.
        """
        normalized_pages = list(dict.fromkeys(int(page) for page in pages))
        if not normalized_pages or not doc_codes:
            return []
        doc_marks = ",".join("?" for _ in doc_codes)
        page_marks = ",".join("?" for _ in normalized_pages)
        sql = f"""
            SELECT c.id AS chunk_id, c.document_id, c.hierarchy, c.page, c.content,
                   d.doc_type, d.doc_code, d.title, d.filename, d.file_path
              FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE d.doc_code IN ({doc_marks})
               AND c.page IN ({page_marks})
             ORDER BY c.page ASC, c.id ASC LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(
                sql,
                [*doc_codes, *normalized_pages, int(limit)],
            ).fetchall()
        return [{**dict(row), "score": 100.0 - index} for index, row in enumerate(rows)]

    def search_document_scopes(self, doc_codes: list[str]) -> list[dict[str, Any]]:
        """Return the main 1.1 application-scope clauses, excluding appendix scopes."""
        if not doc_codes:
            return []
        marks = ",".join("?" for _ in doc_codes)
        sql = f"""
            SELECT c.id AS chunk_id, c.document_id, c.hierarchy, c.page, c.content,
                   d.doc_type, d.doc_code, d.title, d.filename, d.file_path
              FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE d.doc_code IN ({marks})
               AND REPLACE(c.hierarchy, ' ', '') NOT LIKE '%부록%'
               -- Editions differ: some index the clause as
               -- "[CODE] 1 일반사항 > 1.1 적용범위", while others use the
               -- compact "[CODE] 1.1 적용범위" hierarchy.  The normalized
               -- clause number is the stable scope anchor across both forms.
               AND REPLACE(c.hierarchy, ' ', '') LIKE '%1.1적용범위%'
             ORDER BY c.page ASC, c.id ASC
        """
        with self.connect() as connection:
            rows = connection.execute(sql, doc_codes).fetchall()
        return [{**dict(row), "score": 100.0 - index} for index, row in enumerate(rows)]

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT d.*, COUNT(c.id) AS chunk_count
                     FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                    GROUP BY d.id ORDER BY d.doc_type, d.doc_code, d.title"""
            ).fetchall()
        return [dict(row) for row in rows]

    def replace_hazop_rules(self, rules: list[dict[str, Any]], replace: bool = False) -> int:
        """Upsert digital-twin HAZOP rows while preserving the Java table shape."""
        if not rules:
            return 0
        now = datetime.now(UTC).isoformat()
        columns = (
            "no", "scenario_id", "scenario_name", "tag_id", "item_name", "unit",
            "normal_range", "relevance", "guide_word", "cond_text", "threshold_type",
            "threshold_value", "threshold_basis", "threshold_source", "threshold_confidence",
            "compare_dir", "severity", "severity_rank", "risk_scenario",
            "consequence", "emergency_action", "future_measure", "standard_ref", "source",
            "source_url", "updated_at",
        )
        values: list[tuple[Any, ...]] = []
        for rule in rules:
            # Pydantic models intentionally expose only the portable Java
            # columns.  Fill storage-only/default fields here so API clients
            # can submit the same camelCase rows that the digital twin uses.
            normalized = dict(rule)
            normalized["no"] = int(normalized.get("no") or 0)
            normalized["scenario_id"] = str(normalized.get("scenario_id") or "*")
            normalized["tag_id"] = str(normalized.get("tag_id") or "").strip()
            normalized["source"] = str(normalized.get("source") or "digital-twin-hazop")
            normalized["updated_at"] = str(normalized.get("updated_at") or now)
            normalized["severity_rank"] = int(normalized.get("severity_rank") or 1)
            defaults = {
                "scenario_name": "", "item_name": "", "unit": "", "normal_range": "",
                "relevance": "직접관련", "guide_word": "", "cond_text": "",
                "threshold_type": "STATIC", "threshold_basis": "", "threshold_source": "",
                "threshold_confidence": "unverified", "compare_dir": "", "severity": "주의",
                "risk_scenario": "", "consequence": "", "emergency_action": "",
                "future_measure": "", "standard_ref": "",
            }
            for key, default in defaults.items():
                normalized[key] = str(normalized.get(key) or default)
            values.append(tuple(normalized.get(column) for column in columns))
        with self.transaction() as connection:
            if replace:
                connection.execute("DELETE FROM hazop_rules")
            marks = ",".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO hazop_rules ({','.join(columns)}) VALUES ({marks}) "
                "ON CONFLICT(scenario_id, tag_id, no, cond_text) DO UPDATE SET "
                + ",".join(f"{column}=excluded.{column}" for column in columns if column not in {"scenario_id", "tag_id", "no", "cond_text"}),
                values,
            )
        return len(values)

    def list_hazop_rules(
        self,
        scenario_id: str | None = None,
        tag_id: str | None = None,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if scenario_id:
            conditions.append("(scenario_id = ? OR scenario_id = '*')")
            params.append(scenario_id)
        if tag_id:
            conditions.append("tag_id = ?")
            params.append(tag_id)
        if not include_inactive:
            conditions.append("LOWER(COALESCE(relevance, '')) NOT IN ('해당없음', 'inactive', 'none')")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM hazop_rules{where} ORDER BY severity_rank DESC, no ASC, id ASC",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def save_hazop_evaluation(
        self,
        station_id: str,
        scenario_id: str,
        status: str,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO hazop_evaluations (station_id, scenario_id, evaluated_at, status, payload_json, result_json) VALUES (?, ?, ?, ?, ?, ?)",
                (station_id, scenario_id, now, status, json.dumps(payload, ensure_ascii=False, default=str), json.dumps(result, ensure_ascii=False, default=str)),
            )
            return int(cursor.lastrowid)

    def latest_hazop_evaluation(self, station_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM hazop_evaluations WHERE station_id = ? ORDER BY id DESC LIMIT 1",
                (station_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def existing_document_codes(self, codes: list[str]) -> set[str]:
        normalized = list(dict.fromkeys(code.upper() for code in codes if code))
        if not normalized:
            return set()
        marks = ",".join("?" for _ in normalized)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT doc_code FROM documents WHERE doc_code IN ({marks})",
                normalized,
            ).fetchall()
        return {str(row["doc_code"]).upper() for row in rows}

    def suggest_document_codes(self, requested: str, limit: int = 5) -> list[dict[str, str]]:
        requested = requested.upper()
        requested_number = requested[2:] if re.fullmatch(r"[A-Z]{2}\d{3}", requested) else ""
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT doc_code, MIN(title) AS title, MIN(doc_type) AS doc_type
                     FROM documents
                    WHERE doc_code != 'UNKNOWN'
                    GROUP BY doc_code"""
            ).fetchall()
        same_number_matches: list[tuple[float, str, dict[str, str]]] = []
        fuzzy_matches: list[tuple[float, str, dict[str, str]]] = []
        for row in rows:
            code = str(row["doc_code"]).upper()
            similarity = SequenceMatcher(None, requested, code).ratio()
            item = {"doc_code": code, "title": str(row["title"]), "doc_type": str(row["doc_type"])}
            if requested_number and code[2:] == requested_number:
                same_number_matches.append((-similarity, code, item))
            elif similarity >= 0.6:
                fuzzy_matches.append((-similarity, code, item))
        ranked = same_number_matches or fuzzy_matches
        ranked.sort(key=lambda item: item[:2])
        return [item[2] for item in ranked[:limit]]

    def delete_document(self, document_id: int) -> bool:
        with self.transaction() as connection:
            ids = [row["id"] for row in connection.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
            if ids:
                marks = ",".join("?" for _ in ids)
                connection.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", ids)
                connection.execute(f"DELETE FROM chunks_fts_v2 WHERE chunk_id IN ({marks})", ids)
                connection.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({marks})", ids)
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def save_exchange(self, conversation_id: str, question: str, answer: str, citations: list[dict[str, Any]], model: str, intent: str, latency_ms: int) -> int:
        now = datetime.now(UTC).isoformat()
        citations_json = json.dumps(citations, ensure_ascii=False)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
                (conversation_id, now, now),
            )
            connection.executemany(
                "INSERT INTO messages (conversation_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (conversation_id, "user", question, "[]", now),
                    (conversation_id, "assistant", answer, citations_json, now),
                ],
            )
            cursor = connection.execute(
                """INSERT INTO chat_logs
                   (conversation_id, question, answer, model, intent, references_json, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, question, answer, model, intent, citations_json, latency_ms, now),
            )
            return int(cursor.lastrowid)

    def history(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def update_feedback(self, log_id: int, score: int) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute("UPDATE chat_logs SET feedback_score = ? WHERE id = ?", (score, log_id))
            return cursor.rowcount > 0

    def update_exchange_answer(
        self,
        log_id: int,
        conversation_id: str,
        answer: str,
        citations: list[dict[str, Any]],
        model: str,
    ) -> bool:
        """Persist a post-generation review without leaving stale chat history."""
        citations_json = json.dumps(citations, ensure_ascii=False)
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE chat_logs
                      SET answer = ?, model = ?, references_json = ?
                    WHERE id = ? AND conversation_id = ?""",
                (answer, model, citations_json, log_id, conversation_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """UPDATE messages
                          SET content = ?, citations_json = ?
                        WHERE id = (
                            SELECT id FROM messages
                             WHERE conversation_id = ? AND role = 'assistant'
                             ORDER BY id DESC LIMIT 1
                        )""",
                    (answer, citations_json, conversation_id),
                )
            return cursor.rowcount > 0

    def stats(self) -> dict[str, int]:
        with self.connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            vectors = connection.execute(
                "SELECT COUNT(*) FROM chunk_vectors WHERE model = ?", (VECTOR_MODEL,)
            ).fetchone()[0]
            source_rows = connection.execute(
                "SELECT doc_type, COUNT(*) AS count FROM documents WHERE doc_type IN ('NREL', 'HIAD') GROUP BY doc_type"
            ).fetchall()
            hazop_rules = connection.execute("SELECT COUNT(*) FROM hazop_rules").fetchone()[0]
            hazop_evaluations = connection.execute("SELECT COUNT(*) FROM hazop_evaluations").fetchone()[0]
        result = {
            "documents": int(documents),
            "chunks": int(chunks),
            "vectors": int(vectors),
            "hazop_rules": int(hazop_rules),
            "hazop_evaluations": int(hazop_evaluations),
        }
        for row in source_rows:
            result[f"{str(row['doc_type']).lower()}_documents"] = int(row["count"])
        return result
