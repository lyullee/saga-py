from pathlib import Path

from saga.database import Database
from saga.indexer import PdfIndexer


def test_page_chunks_keep_one_line_numbered_rules_as_content():
    text = (
        "2.7.2.3.3 검지부의 설치높이는 가스 비중과 주위 상황에 따라 정한다.\n"
        "2.7.2.3.4 검지부의 설치장소는 관계자가 상주하거나 경보를 식별할 수 있는 장소로써 경보가 울린\n"
        "후 각종 조치를 취하기에 적절한 위치로 한다.\n"
        "2.7.3 다음 설비 기준"
    )
    chunks = PdfIndexer(Database(Path("unused.db")))._page_chunks(text, 64, "FS551")

    height = next(item for item in chunks if "2.7.2.3.3" in item["hierarchy"])
    alarm = next(item for item in chunks if "2.7.2.3.4" in item["hierarchy"])
    assert "2.7.2.3.3 검지부의 설치높이는 가스 비중과 주위 상황에 따라 정한다." in height["content"]
    assert "2.7.2.3.4 검지부의 설치장소는 관계자가 상주하거나 경보를 식별할 수 있는 장소" in alarm["content"]
    assert "후 각종 조치를 취하기에 적절한 위치로 한다." in alarm["content"]
