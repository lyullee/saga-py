"""Digital-twin questions use one LLM completion without the RAG pipeline."""

from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from saga.api import app


class FakeReasoner:
    def __init__(self, answer_text="질문에 대한 직접 답변", stream_text="현재 압력 88 MPa"):
        self.calls = []
        self.answer_text = answer_text
        self.stream_text = stream_text

    @contextmanager
    def use_provider(self, provider):
        self.provider = provider
        yield

    async def answer(self, prompt, **kwargs):
        self.calls.append(("answer", self.provider, prompt, kwargs))
        return self.answer_text

    async def answer_stream(self, prompt, **kwargs):
        self.calls.append(("stream", self.provider, prompt, kwargs))
        midpoint = max(1, len(self.stream_text) // 2)
        await kwargs["on_delta"](self.stream_text[:midpoint])
        await kwargs["on_delta"](self.stream_text[midpoint:])
        return self.stream_text

    async def close(self):
        pass


def test_one_pass_chat_and_stream_skip_review_and_honor_question():
    with TestClient(app) as client:
        reasoner = FakeReasoner()
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )
        app.state.pipeline = SimpleNamespace(run=lambda *args: (_ for _ in ()).throw(
            AssertionError("RAG/review pipeline must not run")))
        direct = client.post("/api/digital-twin/chat/direct", json={
            "message": "압력 변화의 원인은?", "provider": "service_hub",
        })
        assert direct.status_code == 200
        assert direct.json()["answer"] == "질문에 대한 직접 답변"
        assert reasoner.calls[0][2] == "압력 변화의 원인은?"
        assert reasoner.calls[0][3]["disable_reasoning"] is True
        assert reasoner.calls[0][3]["model"] == "llama-3.3-70b"

        streamed = client.post("/api/digital-twin/chat/direct/stream", json={
            "message": "현재 압력은?", "provider": "groq",
        })
        assert streamed.status_code == 200
        assert streamed.text.count("event: token") >= 1
        assert "event: answer" in streamed.text
        assert "88 MPa" not in streamed.text
        assert "입력 데이터에 없는 구체 수치는" in streamed.text
        assert reasoner.calls[1][2] == "현재 압력은?"
        assert reasoner.calls[1][3]["reasoning_effort"] == "none"
        assert reasoner.calls[1][3]["model"] == "qwen/qwen3.8-27b"


def test_main_and_sensor_assistants_have_separate_contracts_and_prompts():
    with TestClient(app) as client:
        reasoner = FakeReasoner()
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )
        app.state.pipeline = SimpleNamespace(run=lambda *args: (_ for _ in ()).throw(
            AssertionError("standalone SAGA RAG pipeline must not run")))

        main = client.post("/api/integrations/digital-twin/main", json={
            "question": "압력 보완을 멈추는 방법은?", "provider": "service_hub",
            "request_kind": "user_query", "context": {
                "station_status": "WARNING",
                "consolidated_response_guidance": {
                    "full_plan_delivered_separately": True,
                },
            },
        })
        sensor = client.post("/api/integrations/digital-twin/sensor/stream", json={
            "sensor_id": "GD-0801", "question": "이 경보의 원인은?", "provider": "groq",
            "request_kind": "user_query", "context": {"value": 100.0},
        })

        assert main.status_code == 200
        assert sensor.status_code == 200
        main_prompt = reasoner.calls[0][2]
        sensor_prompt = reasoner.calls[1][2]
        assert "[DIGITAL_TWIN_MAIN_ASSISTANT]" in main_prompt
        assert "압력 보완을 멈추는 방법은?" in main_prompt
        assert "질문과 무관한 전체 설비 상태 보고로 바꾸지 마세요" in main_prompt
        assert "해당 조치 목록을 반복하지 말고" in main_prompt
        assert reasoner.calls[0][3]["max_tokens"] == 900
        assert "[DIGITAL_TWIN_SENSOR_ASSISTANT]" in sensor_prompt
        assert "GD-0801 전용" in sensor_prompt
        assert "이 경보의 원인은?" in sensor_prompt
        assert "SAGA 문서검색 세션을 사용하지 않습니다" in sensor_prompt
        assert "event: answer" in sensor.text


def test_direct_measurement_guard_keeps_supplied_values_and_removes_invented_values():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            answer_text=(
                "현재 압력은 88 MPa입니다. "
                "냉각 목표는 -40 °C입니다. "
                "10 minutes 동안 대기하세요."
            )
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/digital-twin/chat/direct", json={
            "message": "현재 측정 압력은 88 MPa입니다. 상태를 설명해줘.",
            "provider": "service_hub",
        })

        assert response.status_code == 200
        answer = response.json()["answer"]
        assert "88 MPa" in answer
        assert "-40 °C" not in answer
        assert "10 minutes" not in answer
        assert answer.count("입력 데이터에 없는 구체 수치는") == 1


def test_direct_measurement_guard_preserves_equivalent_units_and_range_spelling():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            answer_text=(
                "The test range was -33 °C to -40 °C and the response took 10 minutes. "
                "An unreported pressure of 12 bar was also observed."
            )
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/integrations/digital-twin/main", json={
            "question": "Summarize the historical observation.",
            "provider": "groq",
            "language": "en",
            "request_kind": "user_query",
            "context": {
                "historical_observation": {
                    "description": (
                        "Hydrogen was pre-chilled between -33oC and -40oC. "
                        "Responders arrived within 10 min."
                    ),
                },
            },
        })

        assert response.status_code == 200
        answer = response.json()["answer"]
        assert "-33 °C to -40 °C" in answer
        assert "10 minutes" in answer
        assert "12 bar" not in answer
        assert answer.count("No precise value was supplied") == 1


def test_direct_measurement_guard_expands_shared_unit_ranges():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            answer_text="The expected operating interval is 5 MPa to 10 MPa."
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/digital-twin/chat/direct", json={
            "message": "The documented operating interval is 5-10 MPa.",
            "provider": "service_hub",
        })

        assert response.status_code == 200
        assert response.json()["answer"] == (
            "The expected operating interval is 5 MPa to 10 MPa."
        )


def test_main_assistant_guard_accepts_value_unit_pairs_derived_from_context():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            answer_text=(
                "GD-0901에서 수소 농도 2.0 vol%_H2가 관측됐고 영향거리는 5.5 m입니다. "
                "최대 과압은 12.4 kPa, 최대 열복사는 5.8 kW/m2입니다. "
                "10 minutes 동안 대기하세요."
            )
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/integrations/digital-twin/main", json={
            "question": "현재 사고를 요약해줘.",
            "provider": "groq",
            "request_kind": "user_query",
            "context": {
                "active_alerts": [{
                    "sensor_id": "GD-0901", "value": 2.0, "unit": "vol%_H2",
                }],
                "impact_results": [{
                    "effect_distance_m": 5.5,
                    "maximum_overpressure_kpa": 12.4,
                    "maximum_heat_flux_kw_m2": 5.8,
                }],
            },
        })

        assert response.status_code == 200
        answer = response.json()["answer"]
        assert "2.0 vol%_H2" in answer
        assert "5.5 m" in answer
        assert "12.4 kPa" in answer
        assert "5.8 kW/m2" in answer
        assert "10 minutes" not in answer
        assert answer.count("입력 데이터에 없는 구체 수치는") == 1
        prompt = reasoner.calls[0][2]
        assert "2 vol%_H2" in prompt
        assert "5.5 m" in prompt
        assert "12.4 kPa" in prompt
        assert "5.8 kW/m2" in prompt


def test_sensor_assistant_guard_accepts_live_process_field_units():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            answer_text=(
                "PT-0901 압력은 69.4 MPa, 온도는 25 °C이며 유량은 13.5 g/s입니다. "
                "재고는 41.2 kg이고 검지기 농도는 850 ppm입니다."
            )
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/integrations/digital-twin/sensor", json={
            "sensor_id": "PT-0901",
            "question": "현재 공정값을 설명해줘.",
            "provider": "groq",
            "request_kind": "user_query",
            "context": {
                "pressure_mpa": 69.4,
                "temperature_c": 25.0,
                "mass_flow_g_s": 13.5,
                "inventory_kg": 41.2,
                "detector_ppm": 850,
            },
        })

        assert response.status_code == 200
        answer = response.json()["answer"]
        assert "69.4 MPa" in answer
        assert "25 °C" in answer
        assert "13.5 g/s" in answer
        assert "41.2 kg" in answer
        assert "850 ppm" in answer
        prompt = reasoner.calls[0][2]
        assert "69.4 MPa" in prompt
        assert "25 °C" in prompt
        assert "13.5 g/s" in prompt
        assert "41.2 kg" in prompt
        assert "850 ppm" in prompt


def test_english_sensor_stream_never_emits_raw_unsupported_measurement():
    with TestClient(app) as client:
        reasoner = FakeReasoner(
            stream_text="Pressure is stable. Wait 10 minutes before restart."
        )
        app.state.reasoner = reasoner
        app.state.settings = SimpleNamespace(
            service_hub_api_key="test-key", groq_api_key="test-key",
            service_hub_direct_model="llama-3.3-70b", groq_direct_model="qwen/qwen3.8-27b",
        )

        response = client.post("/api/integrations/digital-twin/sensor/stream", json={
            "sensor_id": "PT-0901", "question": "What should I do?",
            "provider": "groq", "language": "en", "request_kind": "user_query",
            "context": {"state": "ALERT"},
        })

        assert response.status_code == 200
        assert "10 minutes" not in response.text
        assert "No precise value was supplied" in response.text
        assert "event: answer" in response.text
