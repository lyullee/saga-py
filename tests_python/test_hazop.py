from __future__ import annotations

from types import SimpleNamespace

import pytest

from saga.database import Database
from saga.hazop import HazopEngine
from saga.schemas import DigitalTwinHazopRequest, HazopRuleBatchRequest


def _settings(**overrides):
    values = {
        "context_limit": 12,
        "service_hub_api_key": "",
        "service_hub_model": "gpt-oss-20b",
        "reasoning_effort": "low",
        "answer_length": "standard",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_digital_twin_static_rule_produces_hazop_hit_and_sop(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules(
        [{
            "no": 1,
            "scenario_id": "charging",
            "scenario_name": "충전 이상",
            "tag_id": "P-101",
            "item_name": "충전 압력",
            "threshold_type": "STATIC",
            "threshold_value": 10,
            "compare_dir": ">=",
            "severity": "경보",
            "severity_rank": 3,
            "risk_scenario": "과압",
            "consequence": "배관 손상 또는 누출",
            "emergency_action": "설계된 비상차단과 대피 절차를 수행",
            "future_measure": "원인 조사 후 재가동 승인",
            "standard_ref": "내부 HAZOP H-01",
        }],
    )
    engine = HazopEngine(database, _settings(), None)
    request = DigitalTwinHazopRequest.model_validate({
        "stationId": "ST-01",
        "scenarioId": "charging",
        "readings": [{"tagId": "P-101", "value": 12, "unit": "bar"}],
    })

    result = await engine.evaluate(request)

    assert result.status == "WARNING"
    assert result.hit_count == 1
    assert result.hits[0].tag_id == "P-101"
    assert result.hits[0].effective_threshold == 10
    assert result.sop is not None
    assert result.sop.source_status == "hazop_grounded"
    assert "비상차단" in result.sop.answer
    assert result.sop.priority == "emergency"
    assert result.sop.immediate_actions
    assert result.sop.isolation_evacuation
    assert result.sop.verification_steps
    assert result.sop.restart_requirements
    assert result.sop.records_to_capture
    assert result.sop.escalation
    assert result.evaluation_id is not None


@pytest.mark.asyncio
async def test_delta_rule_without_baseline_is_partial_not_false_normal(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([{
        "tag_id": "T-101",
        "threshold_type": "DELTA_START",
        "threshold_value": 5,
        "compare_dir": ">=",
        "severity": "주의",
    }])
    engine = HazopEngine(database, _settings(), None)

    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "T-101", "value": 20}],
    }))

    assert result.status == "PARTIAL"
    assert not result.hits
    assert any("시작 기준값" in item for item in result.unevaluated)


@pytest.mark.asyncio
async def test_unit_mismatch_is_not_silently_compared(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([{
        "tag_id": "P-UNIT",
        "unit": "MPa",
        "threshold_type": "STATIC",
        "threshold_value": 10,
        "compare_dir": ">=",
    }])
    engine = HazopEngine(database, _settings(), None)

    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "P-UNIT", "value": 12, "unit": "bar"}],
    }))

    assert result.status == "PARTIAL"
    assert not result.hits
    assert any("단위" in item for item in result.data_quality)


@pytest.mark.asyncio
async def test_abs_pressure_notation_matches_mpa_rule(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([{
        "tag_id": "P-ABS", "unit": "MPa", "threshold_type": "STATIC",
        "threshold_value": 10, "compare_dir": ">=",
        "threshold_basis": "시뮬레이터 HAZOP", "threshold_source": "SIM-HZ",
        "threshold_confidence": "derived",
    }])
    result = await HazopEngine(database, _settings(), None).evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "P-ABS", "value": 12, "unit": "MPa_abs"}],
    }))
    assert result.status == "WARNING"
    assert result.hits[0].tag_id == "P-ABS"


@pytest.mark.asyncio
async def test_java_delta_semantics_and_guide_word_deduplication(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([
        {"no": 29, "tag_id": "LV-1", "threshold_type": "DELTA_START", "threshold_value": -2,
         "compare_dir": "이하", "guide_word": "Low", "severity": "주의", "severity_rank": 1},
        {"no": 36, "tag_id": "LV-1", "threshold_type": "DELTA_START", "threshold_value": -1,
         "compare_dir": "이하", "guide_word": "Low-Low", "severity": "긴급", "severity_rank": 3,
         "emergency_action": "비상 대응"},
    ])
    engine = HazopEngine(Database(tmp_path / "hazop.db"), _settings(), None)
    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "LV-1", "value": 8, "baselineStart": 10}],
    }))
    assert result.status == "WARNING"
    assert len(result.hits) == 1
    assert result.hits[0].no == 36
    assert result.hits[0].effective_threshold == 9


@pytest.mark.asyncio
async def test_high_label_is_rejected_as_monitoring_compare_direction(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([{
        "tag_id": "P-HIGH",
        "unit": "MPa",
        "threshold_type": "STATIC",
        "threshold_value": 1.2,
        "compare_dir": "High",
        "threshold_basis": "설계자료",
        "threshold_source": "설계자료.pdf p.1",
        "threshold_confidence": "verified",
    }])
    engine = HazopEngine(database, _settings(), None)
    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "P-HIGH", "value": 1.3, "unit": "MPa"}],
    }))
    assert result.status == "PARTIAL"
    assert not result.hits
    assert any("High/High-High" in item for item in result.threshold_gaps)
    readiness = engine.monitoring_readiness()
    assert readiness["monitoring_ready"] is False
    assert readiness["not_ready_rules"] == 1


@pytest.mark.asyncio
async def test_missing_hazop_row_is_explicit_and_catalog_is_not_claimed_as_evidence(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    engine = HazopEngine(database, _settings(), None)

    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "stationId": "ST-02",
        "readings": [{"tagId": "UNKNOWN-1", "value": 1}],
    }))

    assert result.status == "PARTIAL"
    assert result.missing_rule_tags == ["UNKNOWN-1"]
    assert result.sop is not None
    assert result.sop.source_status == "llm_general"
    assert result.sop.references
    assert all(ref.status == "catalog_only" for ref in result.sop.references)
    assert any("HAZOP 테이블에 없는" in item for item in result.sop.limitations)


@pytest.mark.asyncio
async def test_fully_evaluated_normal_state_is_hazop_grounded(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()
    database.replace_hazop_rules([{
        "tag_id": "P-NORMAL",
        "unit": "bar",
        "threshold_type": "STATIC",
        "threshold_value": 10,
        "compare_dir": ">=",
        "threshold_basis": "설계 정상운전 상한 분석",
        "threshold_source": "승인된 설비 설계자료",
        "threshold_confidence": "verified",
    }])
    engine = HazopEngine(database, _settings(), None)
    result = await engine.evaluate(DigitalTwinHazopRequest.model_validate({
        "readings": [{"tagId": "P-NORMAL", "value": 5, "unit": "bar"}],
    }))
    assert result.status == "NORMAL"
    assert result.sop is not None
    assert result.sop.source_status == "hazop_grounded"
    assert result.sop.priority == "normal"


@pytest.mark.asyncio
async def test_direct_evaluation_uses_transferred_catalog_without_llm(tmp_path):
    database = Database(tmp_path / "hazop.db")
    database.initialize()

    class UnexpectedReasoner:
        async def answer(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("direct HAZOP must not invoke an LLM")

    engine = HazopEngine(database, _settings(service_hub_api_key="configured"), UnexpectedReasoner())
    result = await engine.evaluate_direct(DigitalTwinHazopRequest.model_validate({
        "stationId": "SIM-01",
        "readings": [{"tagId": "PT-1101", "value": 72, "unit": "MPa_abs", "quality": "GOOD"}],
        "hazopRules": [{
            "rule_id": "HZ-081", "sensor_id": "PT-1101", "name": "PCV 차압 증가",
            "operator": ">=", "threshold": 30, "unit": "MPa_abs", "severity": "ALARM",
            "source_id": "SIM-HZ", "emergency_action": "ESD 상태와 현장 대응절차를 확인",
        }],
        "impactResults": [{"calculation_status": "calculated", "node_id": "N11",
                           "pressure_sensor": "PT-1101", "maximum_overpressure_pa": 6100.0},
                          {"calculation_status": "input_unavailable", "node_id": "N22"}],
    }))

    assert result.status == "WARNING"
    assert result.hit_count == 1
    assert result.hits[0].tag_id == "PT-1101"
    assert result.hits[0].rule_id is None  # HZ-081 is an external, non-integer ID
    assert result.sop is not None
    assert "직답 모드" in " ".join(result.sop.limitations)
    assert result.processing_ms is not None
    assert len(result.impact_results) == 1
    assert result.impact_results[0]["pressure_sensor"] == "PT-1101"


def test_java_camel_case_schema_and_batch_request_are_accepted():
    request = DigitalTwinHazopRequest.model_validate({
        "stationId": "ST-03",
        "scenario": "normal",
        "readings": [{"tagId": "P-1", "baselineStart": 5, "baselinePrev": 6, "value": 7}],
    })
    batch = HazopRuleBatchRequest.model_validate({
        "replace": True,
        "rules": [{
            "scenarioId": "normal",
            "tagId": "P-1",
            "thresholdType": "STATIC",
            "thresholdValue": 8,
            "compareDir": ">=",
            "emergencyAction": "격리",
        }],
    })
    assert request.station_id == "ST-03"
    assert request.scenario_id == "normal"
    assert request.readings[0].baseline_start == 5
    assert batch.rules[0].tag_id == "P-1"
    assert batch.rules[0].threshold_value == 8
