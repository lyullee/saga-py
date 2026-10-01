from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

import saga.service_hub_client as client_module
from saga.service_hub_client import ServiceHubReasoner


class ShortReply(BaseModel):
    value: str


def test_structured_call_uses_service_hub_openai_compatibility(monkeypatch):
    client_options = {}
    requests = []

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"검증 완료"}'))]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    def fake_openai(**kwargs):
        client_options.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(client_module, "AsyncOpenAI", fake_openai)
    settings = SimpleNamespace(
        service_hub_api_key="test-key-never-sent",
        service_hub_base_url="https://open.hasa.re.kr/v1/",
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)

    async def run():
        result = await reasoner.structured_model("prompt", ShortReply, "short_reply", "gpt-oss-20b", "low", 512)
        await reasoner.close()
        return result

    result = asyncio.run(run())

    assert result.value == "검증 완료"
    assert client_options["api_key"] == "test-key-never-sent"
    assert client_options["base_url"] == "https://open.hasa.re.kr/v1"
    assert client_options["max_retries"] == 0
    assert requests[0]["model"] == "gpt-oss-20b"
    assert requests[0]["max_tokens"] == 512
    assert requests[0]["reasoning_effort"] == "low"
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "include_reasoning" not in requests[0]


def test_structured_call_falls_back_when_json_mode_is_unsupported(monkeypatch):
    requests = []

    class UnsupportedJsonMode(Exception):
        status_code = 400

        def __str__(self):
            return "response_format json_object is not supported"

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise UnsupportedJsonMode()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"검증 완료"}'))]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    monkeypatch.setattr(client_module, "AsyncOpenAI", lambda **kwargs: FakeClient())
    settings = SimpleNamespace(
        service_hub_api_key="test-key",
        service_hub_base_url="https://open.hasa.re.kr/v1",
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)

    async def run():
        result = await reasoner.structured_model("prompt", ShortReply, "short_reply", "gpt-oss-20b", "low", 512)
        await reasoner.close()
        return result

    result = asyncio.run(run())

    assert result.value == "검증 완료"
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]


def test_service_hub_requests_are_serialized(monkeypatch):
    active = 0
    maximum_active = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    monkeypatch.setattr(client_module, "AsyncOpenAI", lambda **kwargs: FakeClient())
    settings = SimpleNamespace(
        service_hub_api_key="test-key",
        service_hub_base_url="https://open.hasa.re.kr/v1",
        service_hub_model="gpt-oss-120b",
        service_hub_fast_model="gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)

    async def run():
        await asyncio.gather(reasoner.answer("first"), reasoner.answer("second"))
        await reasoner.close()

    asyncio.run(run())
    assert maximum_active == 1


def test_answer_stream_forwards_content_deltas(monkeypatch):
    requests = []

    class FakeStream:
        def __aiter__(self):
            self.items = iter(("첫", " 번째", " 답변"))
            return self

        async def __anext__(self):
            try:
                value = next(self.items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=value))]
            )

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            assert kwargs["stream"] is True
            return FakeStream()

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    monkeypatch.setattr(client_module, "AsyncOpenAI", lambda **kwargs: FakeClient())
    settings = SimpleNamespace(
        service_hub_api_key="test-key",
        service_hub_base_url="https://open.hasa.re.kr/v1",
        service_hub_model="gpt-oss-20b",
        service_hub_fast_model="gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)
    received = []

    async def collect(value: str) -> None:
        received.append(value)

    async def run():
        result = await reasoner.answer_stream("prompt", on_delta=collect)
        direct = await reasoner.answer_stream("direct prompt", model="llama-3.3-70b",
                                              disable_reasoning=True, on_delta=collect)
        await reasoner.close()
        return result, direct

    assert asyncio.run(run()) == ("첫 번째 답변", "첫 번째 답변")
    assert received == ["첫", " 번째", " 답변"] * 2
    assert requests[0]["stream"] is True
    assert requests[1]["model"] == "llama-3.3-70b"
    assert "reasoning_effort" not in requests[1]


def test_manual_provider_selection_routes_all_calls_without_fallback(monkeypatch):
    requests = []
    closed = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.name = "groq" if "groq.com" in kwargs["base_url"] else "service_hub"
            self.chat = SimpleNamespace(completions=self)
            self.models = SimpleNamespace(list=self.list_models)

        async def create(self, **kwargs):
            requests.append((self.name, kwargs["model"]))
            if self.name == "groq" and kwargs["messages"][0]["content"] == "fail":
                raise RuntimeError("Groq unavailable")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        async def list_models(self):
            return SimpleNamespace(data=[SimpleNamespace(id=self.name)])

        async def close(self):
            closed.append(self.name)

    monkeypatch.setattr(client_module, "AsyncOpenAI", FakeClient)
    settings = SimpleNamespace(
        service_hub_api_key="hub-test", service_hub_base_url="https://open.hasa.re.kr/v1",
        service_hub_model="gpt-oss-20b", service_hub_fast_model="gpt-oss-20b",
        groq_api_key="groq-test", groq_base_url="https://api.groq.com/openai/v1",
        groq_model="openai/gpt-oss-20b", groq_fast_model="openai/gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)

    async def run():
        assert await reasoner.answer("default") == "ok"
        with reasoner.use_provider("groq"):
            assert await reasoner.answer("selected") == "ok"
            assert await reasoner.model_ids() == ["groq"]
            try:
                await reasoner.answer("fail")
            except RuntimeError as exc:
                assert str(exc) == "Groq unavailable"
            else:
                raise AssertionError("Groq error must not silently use Service Hub")
        assert await reasoner.answer("restored") == "ok"
        await reasoner.close()

    asyncio.run(run())
    assert requests == [
        ("service_hub", "gpt-oss-20b"), ("groq", "openai/gpt-oss-20b"),
        ("groq", "openai/gpt-oss-20b"), ("service_hub", "gpt-oss-20b"),
    ]
    assert set(closed) == {"service_hub", "groq"}


def test_concurrent_requests_keep_their_selected_provider(monkeypatch):
    requests = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.name = "groq" if "groq.com" in kwargs["base_url"] else "service_hub"
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            await asyncio.sleep(0.01)
            requests.append((self.name, kwargs["messages"][0]["content"]))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        async def close(self):
            pass

    monkeypatch.setattr(client_module, "AsyncOpenAI", FakeClient)
    settings = SimpleNamespace(
        service_hub_api_key="hub-test", service_hub_base_url="https://open.hasa.re.kr/v1",
        service_hub_model="gpt-oss-20b", service_hub_fast_model="gpt-oss-20b",
        groq_api_key="groq-test", groq_base_url="https://api.groq.com/openai/v1",
        groq_model="openai/gpt-oss-20b", groq_fast_model="openai/gpt-oss-20b",
        reasoning_effort="low",
    )
    reasoner = ServiceHubReasoner(settings)

    async def selected(provider, message):
        with reasoner.use_provider(provider):
            await reasoner.answer(message)

    async def run():
        await asyncio.gather(selected("groq", "groq question"), selected("service_hub", "hub question"))
        await reasoner.close()

    asyncio.run(run())
    assert set(requests) == {("groq", "groq question"), ("service_hub", "hub question")}
