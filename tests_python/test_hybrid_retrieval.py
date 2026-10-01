from pathlib import Path

from saga.database import Database
from saga.text import build_search_text


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "hybrid.db")
    database.initialize()
    metadata = {
        "doc_type": "CODE",
        "doc_code": "FP217",
        "title": "고압가스 자동차충전소 기준",
        "filename": "FP217.pdf",
        "file_path": str(tmp_path / "FP217.pdf"),
        "file_hash": "hybrid",
        "page_count": 2,
    }
    database.replace_document(
        metadata,
        [
            {
                "hierarchy": "[FP217] 2.1.1 보호시설과의 거리",
                "page": 1,
                "content": "저장설비는 보호시설까지 표 2.1.1.1에서 정한 안전거리 이상을 유지한다.",
                "search_text": build_search_text("FP217", "보호시설과의 거리", "저장설비 안전거리"),
            },
            {
                "hierarchy": "[FP217] 2.3 저장설비",
                "page": 2,
                "content": "저장탱크는 자동차 충돌로부터 보호하는 조치를 한다.",
                "search_text": build_search_text("FP217", "저장설비", "저장탱크 자동차 충돌 보호"),
            },
        ],
    )
    return database


def test_initialize_builds_vector_index_and_structured_metadata(tmp_path: Path):
    database = _database(tmp_path)
    stats = database.stats()
    assert stats["chunks"] == 2
    assert stats["vectors"] == 2
    row = database.connect().execute(
        "SELECT section_number, content_kind FROM chunks ORDER BY id LIMIT 1"
    ).fetchone()
    assert row["section_number"] == "2.1.1"
    assert row["content_kind"] == "paragraph"


def test_hybrid_search_fuses_lexical_and_vector_candidates(tmp_path: Path):
    database = _database(tmp_path)
    results = database.hybrid_search("저장 탱크 보호시설 안전 거리", 5, ["FP217"], "CODE")
    assert results
    assert results[0]["doc_code"] == "FP217"
    assert results[0]["retrieval_method"] == "hybrid"
    assert "보호시설" in results[0]["content"]
