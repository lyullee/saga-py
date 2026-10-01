from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saga.external_sources import expand_external_query  # noqa: E402
from tools.generate_source_queries import generate  # noqa: E402


class _QueryBankDatabase:
    def list_documents(self):
        return [
            {"doc_type": "CODE", "doc_code": code, "title": f"KGS {code}"}
            for code in ("FS551", "FU671", "FP217", "FP216")
        ] + [
            {
                "doc_type": "LAW",
                "doc_code": code,
                "title": code.removeprefix("LAW-"),
            }
            for code in (
                "LAW-수소경제 육성 및 수소 안전관리에 관한 법률",
                "LAW-고압가스 안전관리법",
                "LAW-액화석유가스의 안전관리 및 사업법",
            )
        ] + [
            {"doc_type": "RULE", "doc_code": code, "title": f"사규 {code}"}
            for code in ("2400-1", "2201-1", "2100-1")
        ]


def test_query_bank_is_exactly_300_and_balanced():
    cases = generate(_QueryBankDatabase())
    assert len(cases) == 300
    assert Counter(case["knowledge_mode"] for case in cases) == {
        "standards": 100,
        "operations": 100,
        "incidents": 100,
    }
    assert len({case["id"] for case in cases}) == 300


def test_external_query_expansion_bridges_korean_and_public_source_terms():
    query = expand_external_query("압축기 유지보수와 충전 횟수를 알려줘", "NREL")
    assert "compressor" in query
    assert "maintenance" in query
    assert "fills" in query
    assert "hydrogen station" in query
