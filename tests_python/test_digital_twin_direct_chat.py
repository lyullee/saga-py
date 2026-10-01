"""Digital-twin questions use one LLM completion without the RAG pipeline."""

from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from saga.api import app


class FakeReasoner:
    def __init__(self):
        self.calls = []

    @contextmanager
    def use_provider(self, provider):
        self.provider = provider
        yield

    async def answer(self, prompt, **kwargs):
        self.calls.append(("answer", self.provider, prompt, kwargs))
        return "질문에 대한 직접 답변"

    async def answer_stream(self, prompt, **kwargs):
        self.calls.append(("stream", self.provider, prompt, kwargs))
        await kwargs["on_delta"]("현재 ")
        await kwargs["on_delta"]("압력 88 MPa")
        return "현재 압력 88 MPa"

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
        assert streamed.text.count("event: token") == 2
        assert "event: answer" in streamed.text
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
            "request_kind": "user_query", "context": {"station_status": "WARNING"},
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
        assert "[DIGITAL_TWIN_SENSOR_ASSISTANT]" in sensor_prompt
        assert "GD-0801 전용" in sensor_prompt
        assert "이 경보의 원인은?" in sensor_prompt
        assert "SAGA 문서검색 세션을 사용하지 않습니다" in sensor_prompt
        assert "event: answer" in sensor.text
