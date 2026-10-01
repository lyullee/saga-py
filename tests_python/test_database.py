from pathlib import Path

from saga.database import Database
from saga.text import build_search_text


def test_document_replace_and_search(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    metadata = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "도시가스 배관 기준",
        "filename": "KGS_FS551.pdf",
        "file_path": str(tmp_path / "KGS_FS551.pdf"),
        "file_hash": "abc",
        "page_count": 12,
    }
    chunks = [
        {
            "hierarchy": "[FS551] 2.1 배관",
            "page": 4,
            "content": "배관과 건축물 사이에는 규정된 이격거리를 유지하여야 한다.",
            "search_text": build_search_text("FS551", "배관", "건축물 이격거리"),
        }
    ]
    document_id = database.replace_document(metadata, chunks)
    assert document_id > 0
    results = database.search("배관 이격거리", 10, ["FS551"], "CODE")
    assert len(results) == 1
    assert results[0]["page"] == 4
    assert results[0]["doc_code"] == "FS551"


def test_document_reindex_preserves_document_identity_and_replaces_chunks(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    metadata = {
        "doc_type": "CODE",
        "doc_code": "FS551",
        "title": "도시가스 배관 기준",
        "filename": "KGS_FS551.pdf",
        "file_path": str(tmp_path / "KGS_FS551.pdf"),
        "file_hash": "before",
        "page_count": 12,
    }
    first_id = database.replace_document(
        metadata,
        [{
            "hierarchy": "[FS551] 2.1 배관",
            "page": 4,
            "content": "이전 배관 문장입니다.",
            "search_text": build_search_text("FS551", "2.1 배관", "이전 배관 문장"),
        }],
    )

    updated_id = database.replace_document(
        {**metadata, "title": "갱신된 기준", "file_hash": "after", "page_count": 13},
        [{
            "hierarchy": "[FS551] 2.2 배관",
            "page": 5,
            "content": "새 페이지의 설치 기준입니다.",
            "search_text": build_search_text("FS551", "2.2 배관", "새 페이지 설치 기준"),
        }],
    )

    assert updated_id == first_id
    assert database.list_documents()[0]["id"] == first_id
    assert database.list_documents()[0]["title"] == "갱신된 기준"
    assert database.search("새 페이지 설치 기준", 10, ["FS551"], "CODE")[0]["page"] == 5
    old_matches = database.search("이전 배관 문장", 10, ["FS551"], "CODE")
    assert all("이전 배관 문장" not in item["content"] for item in old_matches)


def test_conversation_and_feedback(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    log_id = database.save_exchange("c1", "질문", "답변", [], "model", "fact", 10)
    assert database.history("c1") == [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "답변"},
    ]
    assert database.update_feedback(log_id, 1)


def test_document_code_suggestions_prefer_same_number(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    for code in ("FP551", "FS551", "FU551", "FF121"):
        database.replace_document(
            {
                "doc_type": "CODE", "doc_code": code, "title": f"{code} 기준",
                "filename": f"{code}.pdf", "file_path": str(tmp_path / f"{code}.pdf"),
                "file_hash": code, "page_count": 1,
            },
            [],
        )

    assert database.existing_document_codes(["FS551", "FF551"]) == {"FS551"}
    suggestions = database.suggest_document_codes("FF551", limit=3)
    assert [item["doc_code"] for item in suggestions] == ["FP551", "FS551", "FU551"]


def test_document_scope_excludes_appendix_scope(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    metadata = {
        "doc_type": "CODE", "doc_code": "FS551", "title": "배관 기준",
        "filename": "FS551.pdf", "file_path": str(tmp_path / "FS551.pdf"),
        "file_hash": "scope-test", "page_count": 2,
    }
    chunks = [
        {
            "hierarchy": "[FS551] 1 일반사항 > 1.1 적용 범위", "page": 1,
            "content": "이 기준은 가스배관의 시설에 적용한다.",
            "search_text": build_search_text("FS551", "적용범위", "가스배관"),
        },
        {
            "hierarchy": "[FS551] 부록 A > A1 적용범위", "page": 2,
            "content": "이 시험은 특정 PE 이음부에 적용한다.",
            "search_text": build_search_text("FS551", "적용범위", "PE 이음부"),
        },
    ]
    database.replace_document(metadata, chunks)
    second_metadata = {
        **metadata, "doc_code": "FU551", "title": "사용시설 기준",
        "filename": "FU551.pdf", "file_path": str(tmp_path / "FU551.pdf"),
        "file_hash": "scope-test-2",
    }
    database.replace_document(
        second_metadata,
        [
            {
                "hierarchy": "[FU551] 1 일반사항 > 1.1 적용 범위", "page": 1,
                "content": "이 기준은 가스사용시설의 설치·운영 및 검사에 적용한다.",
                "search_text": build_search_text("FU551", "적용범위", "가스사용시설"),
            }
        ],
    )

    results = database.search_document_scopes(["FS551", "FU551"])

    assert {item["doc_code"] for item in results} == {"FS551", "FU551"}
    assert all("부록" not in item["hierarchy"] for item in results)
