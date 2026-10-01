import pytest

from saga import api
from saga.schemas import ChatRequest, ChatResponse, DigitalTwinMainAssistantRequest


@pytest.mark.asyncio
async def test_english_rag_translates_query_then_verified_answer(monkeypatch):
    seen = []

    class Reasoner:
        async def answer(self, prompt, **kwargs):
            seen.append(("translation", prompt))
            return "수소 저장 안전거리 기준은?" if "Translate this English" in prompt else "Check the safety distance [1]."

    class Pipeline:
        async def run(self, request, **kwargs):
            seen.append(("retrieval", request.message))
            return ChatResponse(
                conversation_id="test", log_id=1, answer="안전거리를 확인하세요 [1].",
                citations=[], rewritten_query=request.message, intent="fact",
            )

    monkeypatch.setattr(api.app.state, "reasoner", Reasoner(), raising=False)
    monkeypatch.setattr(api.app.state, "pipeline", Pipeline(), raising=False)
    monkeypatch.setattr(api.app.state, "settings", type("Settings", (), {
        "service_hub_fast_model": "fast", "groq_fast_model": "fast",
    })(), raising=False)
    result = await api._run_localized_chat(ChatRequest(message="What is the hydrogen storage safety distance?", language="en"))
    assert result.language == "en"
    assert result.answer == "Check the safety distance [1]."
    assert seen[1] == ("retrieval", "수소 저장 안전거리 기준은?")
    assert "안전거리를 확인하세요 [1]." in seen[2][1]


@pytest.mark.asyncio
async def test_english_stream_emits_visible_translation_deltas(monkeypatch):
    chunks = []

    class Reasoner:
        async def answer(self, prompt, **kwargs):
            return "수소 안전거리는?"

        async def answer_stream(self, prompt, *, on_delta, **kwargs):
            for part in ("Check ", "the source [1]."):
                await on_delta(part)
            return "Check the source [1]."

    class Pipeline:
        async def run(self, request, **kwargs):
            return ChatResponse(
                conversation_id="test", log_id=1, answer="원문 확인 [1].",
                citations=[], rewritten_query=request.message, intent="fact",
            )

    monkeypatch.setattr(api.app.state, "reasoner", Reasoner(), raising=False)
    monkeypatch.setattr(api.app.state, "pipeline", Pipeline(), raising=False)
    monkeypatch.setattr(api.app.state, "settings", type("Settings", (), {
        "service_hub_fast_model": "fast", "groq_fast_model": "fast",
    })(), raising=False)

    async def collect(part):
        chunks.append(part)

    result = await api._run_localized_chat(
        ChatRequest(message="What is the safe distance?", language="en"), token=collect,
    )
    assert chunks == ["Check ", "the source [1]."]
    assert result.answer == "Check the source [1]."


@pytest.mark.asyncio
async def test_english_external_source_query_keeps_english_retrieval(monkeypatch):
    seen = []

    class Reasoner:
        async def answer(self, prompt, **kwargs):
            seen.append(prompt)
            return "Compressor downtime is discussed in the source [1]."

    class Pipeline:
        async def run(self, request, **kwargs):
            seen.append(request.message)
            return ChatResponse(
                conversation_id="test", log_id=1, answer="압축기 가동중단 근거 [1].",
                citations=[], rewritten_query=request.message, intent="fact",
            )

    monkeypatch.setattr(api.app.state, "reasoner", Reasoner(), raising=False)
    monkeypatch.setattr(api.app.state, "pipeline", Pipeline(), raising=False)
    monkeypatch.setattr(api.app.state, "settings", type("Settings", (), {
        "service_hub_fast_model": "fast", "groq_fast_model": "fast",
    })(), raising=False)
    await api._run_localized_chat(ChatRequest(
        message="How often does the compressor fail?", language="en", knowledge_mode="operations",
    ))
    assert seen[0] == "How often does the compressor fail?"


def test_digital_twin_prompt_uses_selected_language():
    request = DigitalTwinMainAssistantRequest(question="What is the first action?", language="en")
    assert "Answer in English" in api._main_assistant_prompt(request)
    request.language = "ko"
    assert "Answer in English" not in api._main_assistant_prompt(request)
