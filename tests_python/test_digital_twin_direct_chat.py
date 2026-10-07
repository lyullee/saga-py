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
