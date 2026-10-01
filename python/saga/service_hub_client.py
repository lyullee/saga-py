from __future__ import annotations

import asyncio
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import Settings


T = TypeVar("T", bound=BaseModel)


class ServiceHubReasoner:
    """Request-scoped OpenAI-compatible clients with explicit provider selection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = (
            AsyncOpenAI(
                api_key=settings.service_hub_api_key,
                base_url=settings.service_hub_base_url.rstrip("/"),
                timeout=120.0,
                # RAG has explicit model fallbacks; SDK-level retries would repeat
                # a full 120-second timeout before those fallbacks can run.
                max_retries=0,
            )
            if settings.service_hub_api_key
            else None
        )
        groq_key = getattr(settings, "groq_api_key", "")
        self.groq_client = (
            AsyncOpenAI(
                api_key=groq_key,
                base_url=getattr(settings, "groq_base_url", "https://api.groq.com/openai/v1").rstrip("/"),
                timeout=120.0,
                max_retries=0,
            )
            if groq_key else None
        )
        self._provider: ContextVar[str] = ContextVar("saga_llm_provider", default="service_hub")
        # The development key is limited to one concurrent request. Queue this
        # process's calls instead of letting concurrent chats fail with 429s.
        self._request_slots = asyncio.Semaphore(1)
        self._groq_request_slots = asyncio.Semaphore(2)

    @contextmanager
    def use_provider(self, provider: str):
        if provider not in {"service_hub", "groq"}:
            raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
        token = self._provider.set(provider)
        try:
            yield
        finally:
            self._provider.reset(token)

    def _selected_model(self, model: str | None) -> str:
        if self._provider.get() != "groq":
            return model or self.settings.service_hub_model
        primary = getattr(self.settings, "groq_model", "openai/gpt-oss-20b")
        fast = getattr(self.settings, "groq_fast_model", primary)
        if model == self.settings.service_hub_fast_model:
            return fast
        if not model or model == self.settings.service_hub_model:
            return primary
        if model.startswith("openai/gpt-oss-"):
            return model
        if model.startswith("qwen/"):
            return model
        if model.startswith("gpt-oss-"):
            return f"openai/{model}"
        # Service Hub-only model names cannot be sent to Groq.
        return primary

    def _selected_slots(self) -> asyncio.Semaphore:
        return self._groq_request_slots if self._provider.get() == "groq" else self._request_slots

    def _require_client(self) -> AsyncOpenAI:
        if self._provider.get() == "groq":
            if self.groq_client is None:
                raise RuntimeError("GROQ_API_KEY가 설정되지 않았습니다.")
            return self.groq_client
        if self.client is None:
            raise RuntimeError("OPEN_AI_SERVICE_HUB_API_KEY가 설정되지 않았습니다.")
        return self.client

    async def _create_completion(self, **kwargs: Any) -> Any:
        kwargs["model"] = self._selected_model(kwargs.get("model"))
        async with self._selected_slots():
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    return await self._require_client().chat.completions.create(**kwargs)
                except Exception as exc:
                    last_error = exc
                    if getattr(exc, "status_code", None) != 429:
                        raise
                    # Service Hub returns retry_after in both normal RPM and
                    # temporary abuse-limit responses. Waiting here prevents a
                    # burst of planner/reranker/reviewer calls from degrading
                    # into an empty or weak answer.
                    match = re.search(r"retry_after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", str(exc), re.I)
                    delay = float(match.group(1)) if match else min(60.0, 6.0 * (attempt + 1))
                    await asyncio.sleep(max(1.0, min(delay, 90.0)))
            raise last_error or RuntimeError("Service Hub request failed")

    async def structured(self, prompt: str, schema: type[T], name: str) -> T:
        return await self.structured_model(
            prompt, schema, name, self.settings.service_hub_fast_model, "low", 1400
        )

    async def structured_model(
        self,
        prompt: str,
        schema: type[T],
        name: str,
        model: str,
        reasoning_effort: str,
        max_completion_tokens: int,
    ) -> T:
        json_schema = schema.model_json_schema()
        messages = [
            {
                "role": "user",
                "content": (
                    f"다음 지시를 수행하고 {name} JSON 객체 하나만 응답하세요. "
                    "JSON은 아래 스키마를 따라야 합니다. 근거 없이 문서 번호나 사실을 만들지 마세요.\n\n"
                    f"JSON Schema:\n{json.dumps(json_schema, ensure_ascii=False)}\n\n"
                    f"지시:\n{prompt}"
                ),
            }
        ]
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
            "response_format": {"type": "json_object"},
        }
        try:
            completion = await self._create_completion(**request)
        except Exception as exc:
            # Some OpenAI-compatible servers do not implement JSON mode. The
            # schema remains in the prompt and is validated locally in that case.
            error_text = str(exc).lower()
            status_code = getattr(exc, "status_code", None)
            if status_code not in {400, 404, 422} or not any(
                term in error_text for term in ("response_format", "json_object", "json mode")
            ):
                raise
            request.pop("response_format")
            completion = await self._create_completion(**request)
        content = completion.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(content))

    async def answer(
        self,
        prompt: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 5000,
        disable_reasoning: bool = False,
    ) -> str:
        request = dict(
            model=model or self.settings.service_hub_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        if not disable_reasoning:
            request["reasoning_effort"] = reasoning_effort or self.settings.reasoning_effort
        completion = await self._create_completion(**request)
        return (completion.choices[0].message.content or "").strip()

    async def answer_stream(
        self,
        prompt: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 5000,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        disable_reasoning: bool = False,
    ) -> str:
        """Generate an answer and forward each visible content delta immediately.

        Service Hub is OpenAI-compatible, so ``stream=True`` works for models
        that expose chat-completion streaming.  A non-stream fallback keeps the
        application usable with a compatible gateway that rejects the flag.
        """
        request = {
            "model": self._selected_model(model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if not disable_reasoning:
            request["reasoning_effort"] = reasoning_effort or self.settings.reasoning_effort
        try:
            async with self._selected_slots():
                stream = await self._require_client().chat.completions.create(**request)
                parts: list[str] = []
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0].delta, "content", None) or ""
                    if not delta:
                        continue
                    parts.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)
                return "".join(parts).strip()
        except Exception as exc:
            # Some OpenAI-compatible deployments support chat completions but
            # not streaming. Retry once without ``stream`` and still emit the
            # completed text through the same callback.
            error_text = str(exc).lower()
            status_code = getattr(exc, "status_code", None)
            if status_code not in {400, 404, 422} and "stream" not in error_text:
                raise
            request.pop("stream", None)
            completion = await self._create_completion(**request)
            text = (completion.choices[0].message.content or "").strip()
            if text and on_delta is not None:
                await on_delta(text)
            return text

    async def model_ids(self) -> list[str]:
        async with self._selected_slots():
            response = await self._require_client().models.list()
        return sorted(model.id for model in response.data)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
        if self.groq_client is not None:
            await self.groq_client.close()
            self.groq_client = None
