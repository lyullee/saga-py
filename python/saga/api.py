from __future__ import annotations

import asyncio
import json
import re
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from mimetypes import guess_type
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings, get_settings
from .database import Database
from .external_sources import ExternalSourceIngestor
from .hazop import HazopEngine
from .indexer import PdfIndexer
from .law_api import LAW_CATALOG, LawApiIngestor
from .rag import RagPipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    ExternalSourceSyncRequest,
    FeedbackRequest,
    DigitalTwinHazopRequest,
    DigitalTwinHazopResponse,
    DigitalTwinDirectChatRequest,
    DigitalTwinMainAssistantRequest,
    DigitalTwinSensorAssistantRequest,
    HazopRuleBatchRequest,
    HazopRuleInput,
    HazopThresholdProposalRequest,
    HazopThresholdProposalResponse,
    LawSyncRequest,
)
from .service_hub_client import ServiceHubReasoner


UI_DIR = PROJECT_ROOT / "python" / "saga" / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = Database(settings.database_path)
    database.initialize()
    app.state.settings = settings
    app.state.database = database
    app.state.indexer = PdfIndexer(database)
    app.state.external_sources = ExternalSourceIngestor(
        database, app.state.indexer, settings.external_data_dir
    )
    reasoner = ServiceHubReasoner(settings)
    app.state.reasoner = reasoner
    app.state.pipeline = RagPipeline(settings, database, reasoner)
    app.state.hazop = HazopEngine(database, settings, reasoner, app.state.pipeline)
    try:
        yield
    finally:
        await reasoner.close()


app = FastAPI(
    title="SAGA Safety AI Governance Agent",
    version="2.0.0",
    description="Open AI Service Hub 추론 모델 기반 KGS 규정 검색·대화 시스템",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")


def settings_dependency() -> Settings:
    return app.state.settings


def require_admin(
    x_admin_token: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
) -> None:
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="관리자 토큰이 필요합니다.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    settings: Settings = app.state.settings
    return {
        "status": "ready" if settings.service_hub_api_key or settings.groq_api_key else "configuration_required",
        "service_hub_configured": bool(settings.service_hub_api_key),
        "groq_configured": bool(settings.groq_api_key),
        "base_url": settings.service_hub_base_url,
        "model": settings.service_hub_model,
        "providers": {
            "service_hub": bool(settings.service_hub_api_key),
            "groq": bool(settings.groq_api_key),
        },
        **app.state.database.stats(),
    }


@app.get("/api/config")
async def public_config() -> dict:
    settings: Settings = app.state.settings
    return {
        "model": settings.service_hub_model,
        "reasoning_effort": settings.reasoning_effort,
        "retrieval_limit": settings.retrieval_limit,
        "context_limit": settings.context_limit,
        "hybrid_search_enabled": settings.hybrid_search_enabled,
        "vector_model": settings.vector_model,
        "answer_review_enabled": settings.answer_review_enabled,
        "law_api_configured": bool(settings.law_api_oc),
        "law_catalog": LAW_CATALOG,
        "external_sources": {
            "nrel": "NREL 운전·고장 통계",
            "hiad": "HIAD 수소 사고 사례",
        },
        "hazop": {
            "rules": app.state.database.stats().get("hazop_rules", 0),
            "evaluate_endpoint": "/api/digital-twin/hazop/evaluate",
            "rule_import_endpoint": "/api/hazop/rules/import",
            "readiness_endpoint": "/api/hazop/rules/monitoring-readiness",
            "proposal_endpoint": "/api/hazop/rules/propose-threshold",
        },
    }


@app.get("/api/models")
async def models(provider: Literal["service_hub", "groq"] = "service_hub") -> dict:
    """Return models for the manually selected chat provider."""
    settings: Settings = app.state.settings
    try:
        with app.state.reasoner.use_provider(provider):
            available = await app.state.reasoner.model_ids()
    except Exception:
        available = []
    blocked_markers = (
        "image", "t2v", "i2v", "ltx", "vision", "-vl", "_vl", "vl-",
        "whisper", "melotts", "ocr", "embed", "bge", "reranker", "cuquantum",
        "pii", "safety", "guardian", "omni",
    )
    available = [
        model for model in available
        if not any(marker in model.lower() for marker in blocked_markers)
        and (provider != "groq" or model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
    ]
    fallback = ([settings.groq_model, settings.groq_fast_model] if provider == "groq" else [
        settings.service_hub_model, settings.service_hub_fast_model, settings.service_hub_vision_model,
    ])
    fallback = [
        model for model in fallback
        if not any(marker in model.lower() for marker in blocked_markers)
    ]
    model_ids = list(dict.fromkeys([*available, *fallback]))
    return {"models": model_ids, "selected": settings.groq_model if provider == "groq" else settings.service_hub_model,
            "configured": bool(settings.groq_api_key if provider == "groq" else settings.service_hub_api_key)}


@app.get("/api/hazop/rules", response_model=list[HazopRuleInput])
async def hazop_rules(
    scenario_id: str | None = Query(default=None, alias="scenarioId"),
    tag_id: str | None = Query(default=None, alias="tagId"),
    scenario_id_snake: str | None = Query(default=None, alias="scenario_id"),
    tag_id_snake: str | None = Query(default=None, alias="tag_id"),
) -> list[HazopRuleInput]:
    """Inspect the HAZOP rows currently used by digital-twin evaluation."""
    rows = await asyncio.to_thread(
        app.state.database.list_hazop_rules,
        scenario_id or scenario_id_snake,
        tag_id or tag_id_snake,
        False,
    )
    return [HazopRuleInput.model_validate(row) for row in rows]


@app.get("/api/hazop/rules/monitoring-readiness")
async def hazop_monitoring_readiness(
    scenario_id: str | None = Query(default=None, alias="scenarioId"),
    scenario_id_snake: str | None = Query(default=None, alias="scenario_id"),
) -> dict:
    """Audit numeric thresholds and evidence before enabling live monitoring."""
    return await asyncio.to_thread(
        app.state.hazop.monitoring_readiness,
        scenario_id or scenario_id_snake,
    )


@app.post("/api/hazop/rules/propose-threshold", response_model=HazopThresholdProposalResponse)
async def propose_hazop_threshold(
    request: HazopThresholdProposalRequest,
) -> HazopThresholdProposalResponse:
    """Propose an evidence-cited numeric threshold; never activates it automatically."""
    return await app.state.hazop.propose_threshold(request)


@app.post("/api/hazop/rules/import", dependencies=[Depends(require_admin)])
async def import_hazop_rules(request: HazopRuleBatchRequest) -> dict:
    """Import Java-compatible HAZOP rows from the digital-twin service."""
    rows = [item.model_dump(by_alias=False) for item in request.rules]
    imported = await asyncio.to_thread(
        app.state.database.replace_hazop_rules,
        rows,
        request.replace,
    )
    readiness = await asyncio.to_thread(app.state.hazop.monitoring_readiness)
    return {
        "imported": imported,
        "replace": request.replace,
        "total": len(await asyncio.to_thread(app.state.database.list_hazop_rules)),
        "monitoring_readiness": readiness,
    }


@app.post("/api/digital-twin/hazop/evaluate", response_model=DigitalTwinHazopResponse)
@app.post("/api/hazop/evaluate-state", response_model=DigitalTwinHazopResponse, include_in_schema=False)
async def evaluate_digital_twin_state(request: DigitalTwinHazopRequest) -> DigitalTwinHazopResponse:
    """Evaluate a digital-twin snapshot against HAZOP and indexed RAG evidence."""
    try:
        return await app.state.hazop.evaluate(request)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"HAZOP 상태 평가 실패: {exc}") from exc


def _digital_twin_direct_model(provider: str) -> tuple[str, str | None]:
    settings = app.state.settings
    if provider == "groq":
        model = settings.groq_direct_model
        return model, "none" if model.startswith("qwen/") else "low"
    return settings.service_hub_direct_model, None


def _digital_twin_provider_ready(provider: str) -> None:
    settings = app.state.settings
    if not (settings.groq_api_key if provider == "groq" else settings.service_hub_api_key):
        raise HTTPException(status_code=503, detail=f"선택한 제공자({provider})의 API 키가 설정되지 않았습니다.")


_DIRECT_CONTEXT_UNIT_SUFFIXES = (
    ("_kw_m2", "kW/m2"),
    ("_w_m2", "W/m2"),
    ("_mpa_abs", "MPa"),
    ("temperature_c", "°C"),
    ("temp_c", "°C"),
    ("_volpct_h2", "vol%_H2"),
    ("_diameter_mm", "mm"),
    ("_duration_s", "s"),
    ("_distance_m", "m"),
    ("_radius_m", "m"),
    ("_mpa", "MPa"),
    ("_kpa", "kPa"),
    ("_pa", "Pa"),
    ("_bar", "bar"),
    ("_kg_s", "kg/s"),
    ("_g_s", "g/s"),
    ("_degc", "°C"),
    ("_ppm", "ppm"),
    ("_time_s", "s"),
    ("_kg", "kg"),
    ("_percent", "%"),
    ("_pct", "%"),
)


def _direct_measurement_evidence(context: object) -> str:
    """Project structured live values into explicit value/unit pairs.

    API contexts often encode a unit in a key such as ``effect_distance_m``
    or in a sibling ``unit`` field. The safety guard needs the equivalent
    visible pair (``5.5 m``) so it does not reject a supported model sentence.
    """

    pairs: list[str] = []

    def append(value: object, unit: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if not isinstance(unit, str) or not unit.strip():
            return
        pairs.append(f"{value:g} {unit.strip()}")

    def visit(value: object) -> None:
        if isinstance(value, dict):
            append(value.get("value"), value.get("unit"))
            for key, child in value.items():
                if isinstance(child, bool):
                    continue
                if isinstance(child, (int, float)):
                    normalized = str(key).lower()
                    for suffix, unit in _DIRECT_CONTEXT_UNIT_SUFFIXES:
                        if normalized.endswith(suffix):
                            append(child, unit)
                            break
                elif isinstance(child, (dict, list, tuple)):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(context)
    return "; ".join(dict.fromkeys(pairs)) or "(없음)"


def _main_assistant_prompt(request: DigitalTwinMainAssistantRequest) -> str:
    history = "\n".join(f"{turn.role}: {turn.content}" for turn in request.history)[-1800:]
    if request.request_kind == "user_query":
        mode = (
            "사용자가 방금 입력한 질문이나 지시를 최우선으로 처리하세요. 첫 문장에서 바로 답하고, "
            "질문과 무관한 전체 설비 상태 보고로 바꾸지 마세요. 조작을 요청받아도 실제 실행되지 않은 "
            "동작을 완료했다고 말하지 말고, 제공된 실행 결과가 있을 때만 완료로 표현하세요. "
            "답변은 질문에 필요한 결론과 근거만 2~4문장으로 제시하세요. 단계별 조치를 물으면 "
            "가장 먼저 할 일과 그 다음 확인할 조건만 간결하게 쓰고, 전체 사고 시나리오와 "
            "상세 계산값을 일괄 나열하지 마세요. context의 consolidated_response_guidance는 "
            "상세 계획이 화면에 별도로 제공된다는 표지이므로 해당 조치 목록을 반복하지 말고 "
            "질문에 필요한 최우선 행동만 답하세요. 상세 계산과 절차는 화면의 별도 영역에 표시됩니다."
        )
    else:
        mode = "자동 분석 요청입니다. 현재 이상과 운전자 조치 우선순위를 짧게 보고하세요."
    return (
        "[DIGITAL_TWIN_MAIN_ASSISTANT]\n"
        "당신은 가상 수소충전소 메인 운전 화면의 직답형 보조자입니다. 추론 과정이나 내부 규칙명은 "
        "노출하지 않습니다. 제공된 계산값만 사용하고 피해영향 재계산을 사용자에게 요구하지 마세요. "
        f"{mode} "
        + ("Answer in English. Keep sensor tags, numerical values, and Korean source names exact.\n\n"
           if request.language == "en" else "\n\n")
        + f"사용자 요청:\n{request.question}\n\n"
        f"최근 대화(현재 데이터보다 우선하지 않음):\n{history or '(없음)'}\n\n"
        "허용된 계산값(아래 값과 단위만 그대로 인용):\n"
        + _direct_measurement_evidence(request.context)
        + "\n\n"
        "현재 계산 데이터(JSON):\n"
        + json.dumps(request.context, ensure_ascii=False, default=str)[:7000]
    )[:10000]


def _sensor_assistant_prompt(request: DigitalTwinSensorAssistantRequest) -> str:
    question = request.question.strip() or f"{request.sensor_id}의 현재 상태와 주의점을 분석해줘."
    if request.request_kind == "user_query":
        mode = (
            "사용자의 추가 질문 또는 지시가 핵심입니다. 첫 문장에서 그것에 직접 답하세요. 센서 상태는 "
            "답의 근거로 필요한 만큼만 사용하고, 고정된 상태 보고서로 질문을 대체하지 마세요. 조작을 "
            "요청받아도 context에 실행 결과가 없으면 완료했다고 주장하지 마세요."
        )
    else:
        mode = "센서를 처음 열어 수행하는 자동 분석입니다. 현재 판정과 필요한 조치를 간결하게 보고하세요."
    return (
        "[DIGITAL_TWIN_SENSOR_ASSISTANT]\n"
        f"당신은 {request.sensor_id} 전용 직답형 센서 분석 보조자입니다. 다른 화면의 대화나 SAGA 문서검색 "
        "세션을 사용하지 않습니다. 실제 신호, 경보 이력, 모의 누출, 가정 누출을 구분하고 제공된 피해영향 "
        f"결과만 해석하세요. {mode} "
        + ("Answer in English. Keep sensor tags, numerical values, and Korean source names exact.\n\n"
           if request.language == "en" else "\n\n")
        + f"사용자 요청:\n{question}\n\n"
        "허용된 계산값(아래 값과 단위만 그대로 인용):\n"
        + _direct_measurement_evidence(request.context)
        + "\n\n"
        "선택 센서 계산 데이터(JSON):\n"
        + json.dumps(request.context, ensure_ascii=False, default=str)[:7600]
    )[:10000]


_DIRECT_MEASUREMENT_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<first>[+-]?(?:\d+(?:[.,]\d+)?|\.\d+))"
    r"(?:\s*(?:-|~|–|—|to|부터|에서)\s*"
    r"(?P<second>[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)))?"
    r"\s*(?P<unit>"
    r"kW\s*/\s*m(?:\^?2|²)|W\s*/\s*m(?:\^?2|²)|"
    r"kg\s*/\s*s|g\s*/\s*s|kg\s*/\s*h|kg\s*/\s*min|"
    r"vol\s*%\s*(?:_?\s*H2)?|ppm|"
    r"MPa|kPa|Pa|bar|"
    r"°\s*C|℃|[oº]\s*C|deg\s*C|degrees?\s+C(?:elsius)?|C|"
    r"millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|mm|cm|m|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|s|min|h|"
    r"kilograms?|grams?|kg|g|%"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _canonical_measurement(value: str) -> str:
    normalized = value.strip().lower().replace(",", ".")
    normalized = re.sub(
        r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)",
        lambda match: f"{float(match.group(0)):.12g}",
        normalized,
    )
    normalized = normalized.replace("℃", "°c").replace("²", "2")
    normalized = re.sub(r"(?:°|º|o)\s*c\b", "°c", normalized)
    normalized = re.sub(r"(?<=\d)\s*c\b", "°c", normalized)
    normalized = re.sub(r"degrees?\s+c(?:elsius)?", "°c", normalized)
    normalized = re.sub(r"deg\s*c", "°c", normalized)
    normalized = re.sub(r"\bseconds?\b|\bsecs?\b", "s", normalized)
    normalized = re.sub(r"\bminutes?\b|\bmins?\b", "min", normalized)
    normalized = re.sub(r"\bhours?\b|\bhrs?\b", "h", normalized)
    normalized = re.sub(r"\bmillimet(?:er|re)s?\b", "mm", normalized)
    normalized = re.sub(r"\bcentimet(?:er|re)s?\b", "cm", normalized)
    normalized = re.sub(r"\bmet(?:er|re)s?\b", "m", normalized)
    normalized = re.sub(r"\bkilograms?\b", "kg", normalized)
    normalized = re.sub(r"\bgrams?\b", "g", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _measurement_atoms(text: str) -> set[str]:
    """Return canonical value/unit atoms, expanding a shared-unit range.

    Treating a range as one opaque string caused grounded values to be removed
    when the source used ``5-10 MPa`` and the answer used ``5 MPa to 10 MPa``.
    Atomic comparison also makes equivalent time and temperature spellings
    compare consistently.
    """

    atoms: set[str] = set()
    for match in _DIRECT_MEASUREMENT_RE.finditer(text):
        unit = match.group("unit")
        atoms.add(_canonical_measurement(f"{match.group('first')} {unit}"))
        if match.group("second") is not None:
            atoms.add(_canonical_measurement(f"{match.group('second')} {unit}"))
    return atoms


def _guard_direct_measurements(answer: str, allowed_text: str, language: str | None = None) -> str:
    """Remove precise value/unit claims absent from the supplied direct prompt.

    Direct digital-twin responses intentionally have no RAG review pass.  This
    guard therefore treats the serialized live context as the numeric source of
    truth.  It removes the complete sentence containing an unsupported
    measurement instead of silently changing a number.
    """
    if not answer:
        return answer
    allowed = _measurement_atoms(allowed_text)
    resolved_language = language or ("ko" if re.search(r"[가-힣]", allowed_text + answer) else "en")
    notice = (
        "입력 데이터에 없는 구체 수치는 제시하지 않습니다. 현장 계측값과 적용 기준을 확인하세요."
        if resolved_language == "ko"
        else "No precise value was supplied for this point; verify the live measurement and applicable procedure."
    )
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    cleaned: list[str] = []
    inserted_notice = False
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        measurements = _measurement_atoms(stripped)
        if measurements - allowed:
            if not inserted_notice:
                cleaned.append(notice)
                inserted_notice = True
            continue
        cleaned.append(stripped)
    return "\n\n".join(cleaned)


def _guarded_stream_chunks(answer: str, max_chars: int = 56) -> list[str]:
    """Split a validated answer into display-sized chunks for a typing effect."""
    if not answer:
        return []
    chunks: list[str] = []
    for paragraph in answer.splitlines(keepends=True):
        remaining = paragraph
        while len(remaining) > max_chars:
            boundary = max(
                remaining.rfind(" ", 0, max_chars + 1),
                remaining.rfind(". ", 0, max_chars + 1) + 1,
                remaining.rfind("다. ", 0, max_chars + 1) + 2,
            )
            if boundary <= 0:
                boundary = max_chars
            chunks.append(remaining[:boundary])
            remaining = remaining[boundary:]
        if remaining:
            chunks.append(remaining)
    return chunks


async def _isolated_direct_answer(
    prompt: str, provider: str, max_tokens: int, language: str | None = None,
) -> dict:
    _digital_twin_provider_ready(provider)
    model, effort = _digital_twin_direct_model(provider)
    try:
        with app.state.reasoner.use_provider(provider):
            answer = await app.state.reasoner.answer(
                prompt, model=model, reasoning_effort=effort, max_tokens=max_tokens,
                disable_reasoning=effort is None,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"직답 LLM 호출 실패: {exc}") from exc
    answer = _guard_direct_measurements(answer, prompt, language)
    return {"answer": answer, "model": model, "provider": provider}


def _isolated_direct_stream(
    prompt: str, provider: str, max_tokens: int, language: str | None = None,
) -> StreamingResponse:
    _digital_twin_provider_ready(provider)
    model, effort = _digital_twin_direct_model(provider)

    async def events():
        async def on_delta(text: str) -> None:
            # Buffer provider tokens.  Emitting them before validation would
            # briefly expose an invented measurement even if the final answer
            # were corrected moments later.
            return None

        async def run() -> str:
            with app.state.reasoner.use_provider(provider):
                return await app.state.reasoner.answer_stream(
                    prompt, model=model, reasoning_effort=effort, max_tokens=max_tokens,
                    on_delta=on_delta, disable_reasoning=effort is None,
                )

        try:
            answer = _guard_direct_measurements(await run(), prompt, language)
            for delta in _guarded_stream_chunks(answer):
                yield f"event: token\ndata: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
            payload = {"answer": answer, "model": model, "provider": provider}
            yield f"event: answer\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/integrations/digital-twin/main")
async def digital_twin_main_assistant(request: DigitalTwinMainAssistantRequest) -> dict:
    """Main-monitor contract; independent from sensor analysis and SAGA RAG chat."""
    return await _isolated_direct_answer(
        _main_assistant_prompt(request), request.provider, request.max_tokens, request.language
    )


@app.post("/api/integrations/digital-twin/main/stream")
async def digital_twin_main_assistant_stream(
    request: DigitalTwinMainAssistantRequest,
) -> StreamingResponse:
    return _isolated_direct_stream(
        _main_assistant_prompt(request), request.provider, request.max_tokens, request.language
    )


@app.post("/api/integrations/digital-twin/sensor")
async def digital_twin_sensor_assistant(request: DigitalTwinSensorAssistantRequest) -> dict:
    """Selected-sensor contract; independent from main-monitor and SAGA RAG chat."""
    return await _isolated_direct_answer(
        _sensor_assistant_prompt(request), request.provider, request.max_tokens, request.language
    )


@app.post("/api/integrations/digital-twin/sensor/stream")
async def digital_twin_sensor_assistant_stream(
    request: DigitalTwinSensorAssistantRequest,
) -> StreamingResponse:
    return _isolated_direct_stream(
        _sensor_assistant_prompt(request), request.provider, request.max_tokens, request.language
    )


@app.post("/api/digital-twin/chat/direct")
async def digital_twin_direct_chat(request: DigitalTwinDirectChatRequest) -> dict:
    """Answer the supplied question once, without RAG planning or review."""
    return await _isolated_direct_answer(
        request.message, request.provider, request.max_tokens
    )


@app.post("/api/digital-twin/chat/direct/stream")
async def digital_twin_direct_chat_stream(request: DigitalTwinDirectChatRequest) -> StreamingResponse:
    """Stream the one-pass answer; never run the multi-stage chat pipeline."""
    return _isolated_direct_stream(
        request.message, request.provider, request.max_tokens
    )


@app.get("/api/digital-twin/state/latest")
async def latest_digital_twin_state(
    station_id: str | None = Query(default=None, alias="stationId"),
    station_id_snake: str | None = Query(default=None, alias="station_id"),
) -> dict:
    resolved_station_id = station_id or station_id_snake or "default"
    row = await asyncio.to_thread(app.state.database.latest_hazop_evaluation, resolved_station_id)
    if row is None:
        return {"station_id": resolved_station_id, "empty": True, "result": None}
    return {"station_id": resolved_station_id, "empty": False, **row}


async def _run_localized_chat(request: ChatRequest, progress=None, token=None) -> ChatResponse:
    if request.language != "en":
        return await app.state.pipeline.run(request, progress=progress, token=token)

    settings: Settings = app.state.settings
    model = settings.groq_fast_model if request.provider == "groq" else settings.service_hub_fast_model
    # The standards library is Korean; NREL/HIAD source libraries are English.
    if request.mode == "rag" and request.knowledge_mode == "standards":
        translated_query = await app.state.reasoner.answer(
            "Translate this English safety question into Korean for retrieval against Korean technical "
            "standards. Preserve KGS codes, equipment tags, units, numbers, and the user's intent. "
            "Return only the Korean question.\n\n" + request.message,
            model=model, max_tokens=500, disable_reasoning=True,
        )
        if not translated_query.strip():
            raise RuntimeError("The English question could not be prepared for Korean document search.")
    else:
        translated_query = request.message
    if progress is not None:
        await progress("Searching source documents…" if request.mode == "rag" else "Preparing the answer…")
    korean_request = request.model_copy(update={"message": translated_query.strip(), "language": "ko"})
    verified = await app.state.pipeline.run(korean_request, progress=None, token=None)
    if progress is not None:
        await progress("Rendering the verified answer in English…")
    english_prompt = (
        "Translate the following verified Korean safety answer into clear English. Preserve every "
        "citation marker such as [1], document code, sensor tag, number, unit, condition, and "
        "uncertainty exactly. Do not add legal or safety claims. Korean quotations remain available "
        "separately in the original citation cards. Return only the English answer.\n\n" + verified.answer
    )
    if token is not None:
        english_answer = await app.state.reasoner.answer_stream(
            english_prompt, model=model, max_tokens=6000,
            on_delta=token, disable_reasoning=True,
        )
    else:
        english_answer = await app.state.reasoner.answer(
            english_prompt, model=model, max_tokens=6000, disable_reasoning=True,
        )
    if not english_answer.strip():
        raise RuntimeError("The verified answer could not be rendered in English.")
    return verified.model_copy(update={
        "answer": english_answer.strip(), "final_answer": english_answer.strip(),
        "draft_answer": "", "review_applied": False, "language": "en",
    })


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not (app.state.settings.groq_api_key if request.provider == "groq" else app.state.settings.service_hub_api_key):
        raise HTTPException(status_code=503, detail=f"선택한 제공자({request.provider})의 API 키가 설정되지 않았습니다.")
    try:
        with app.state.reasoner.use_provider(request.provider):
            return await _run_localized_chat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{request.provider} 처리 중 오류가 발생했습니다. 서버 로그를 확인하세요.") from exc


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Send pipeline stages and actual Service Hub content deltas over SSE."""
    if not (app.state.settings.groq_api_key if request.provider == "groq" else app.state.settings.service_hub_api_key):
        raise HTTPException(status_code=503, detail=f"선택한 제공자({request.provider})의 API 키가 설정되지 않았습니다.")

    async def events():
        event_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        draft_parts: list[str] = []

        async def progress(message: str) -> None:
            await event_queue.put(("status", {"text": message}))

        async def token(text: str) -> None:
            if text:
                # ``token`` is the first-pass answer.  Keep this stream
                # separate from the final reviewed answer so the browser can
                # append the latter instead of replacing the former.
                draft_parts.append(text)
                await event_queue.put(("draft_token", {"text": text}))

        await event_queue.put(("draft_start", {"text": "Preparing the English answer…" if request.language == "en" else "초안 답변을 실시간으로 작성하고 있습니다…"}))
        async def run_selected():
            with app.state.reasoner.use_provider(request.provider):
                return await _run_localized_chat(request, progress=progress, token=token)

        task = asyncio.create_task(run_selected())
        try:
            while True:
                if task.done() and event_queue.empty():
                    break
                try:
                    event_name, payload = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                yield f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            response = await task
            draft_answer = "".join(draft_parts).strip()
            payload = response.model_dump()
            payload.update(
                {
                    "draft_answer": draft_answer,
                    "final_answer": response.answer,
                    "review_applied": bool(
                        draft_answer and draft_answer.strip() != response.answer.strip()
                    ),
                }
            )
            final_status = "Displaying the English answer…" if request.language == "en" else "추론형 검증 결과를 정리하고 있습니다…"
            yield f"event: final_start\ndata: {json.dumps({'text': final_status}, ensure_ascii=False)}\n\n"
            yield f"event: answer\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            if not task.done():
                task.cancel()
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/log/{log_id}/feedback")
async def feedback(log_id: int, request: FeedbackRequest) -> dict:
    if not app.state.database.update_feedback(log_id, request.score):
        raise HTTPException(status_code=404, detail="대화 로그를 찾을 수 없습니다.")
    return {"updated": True}


@app.get("/api/documents", response_model=list[DocumentInfo])
async def documents() -> list[DocumentInfo]:
    return [DocumentInfo(**item) for item in app.state.database.list_documents()]


@app.get("/api/laws/catalog")
async def law_catalog() -> dict:
    return {"laws": LAW_CATALOG, "configured": bool(app.state.settings.law_api_oc)}


@app.post("/api/laws/sync", dependencies=[Depends(require_admin)])
async def sync_laws(request: LawSyncRequest) -> dict:
    """Fetch selected laws, render local PDFs, and index them like any PDF."""
    settings: Settings = app.state.settings
    if not settings.law_api_oc:
        raise HTTPException(
            status_code=503,
            detail="SAGA_LAW_API_OC가 설정되지 않았습니다. 국가법령정보센터 Open API 인증값을 등록하세요.",
        )
    keys = request.keys or list(LAW_CATALOG)
    unknown = [key for key in keys if key not in LAW_CATALOG]
    if unknown:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 법령 키입니다: {', '.join(unknown)}")

    def work() -> list[dict]:
        ingestor = LawApiIngestor(settings.law_api_oc, settings.law_api_base_url, settings.law_api_timeout)
        results: list[dict] = []
        for key in keys:
            try:
                rendered = ingestor.fetch_and_render(key, settings.law_pdf_dir, force=request.force)
                indexed = app.state.indexer.index_pdf(rendered.pdf_path, force=request.force)
                results.append({
                    "key": key,
                    "name": rendered.name,
                    "pdf": str(rendered.pdf_path),
                    "effective_date": rendered.effective_date,
                    "index": asdict(indexed),
                })
            except Exception as exc:
                results.append({"key": key, "name": LAW_CATALOG[key], "status": "failed", "error": str(exc)})
        return results

    results = await asyncio.to_thread(work)
    return {
        "total": len(results),
        "indexed": sum(item.get("index", {}).get("status") == "indexed" for item in results),
        "unchanged": sum(item.get("index", {}).get("status") == "unchanged" for item in results),
        "failed": [item for item in results if item.get("status") == "failed" or item.get("index", {}).get("status") == "failed"],
        "results": results,
    }


@app.post("/api/sources/sync", dependencies=[Depends(require_admin)])
async def sync_external_sources(request: ExternalSourceSyncRequest) -> dict:
    """Download and index the isolated NREL/HIAD corpora."""
    results = await asyncio.to_thread(
        app.state.external_sources.sync,
        request.sources,
        request.force,
    )
    return {
        "total": len(results),
        "indexed": sum(item.get("status") in {"indexed", "unchanged"} for item in results),
        "failed": [item for item in results if item.get("status") == "failed"],
        "results": results,
    }


@app.post("/api/documents/upload", dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
    safe_name = Path(file.filename).name
    destination = (app.state.settings.upload_dir / safe_name).resolve()
    if destination.parent != app.state.settings.upload_dir.resolve():
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    result = await asyncio.to_thread(app.state.indexer.index_pdf, destination, True)
    if result.status == "failed":
        raise HTTPException(status_code=422, detail=result.error)
    return asdict(result)


@app.post("/api/documents/reindex", dependencies=[Depends(require_admin)])
async def reindex(force: bool = False) -> dict:
    roots = [app.state.settings.upload_dir]
    law_root = app.state.settings.law_pdf_dir.resolve()
    if law_root not in {root.resolve() for root in roots}:
        roots.append(app.state.settings.law_pdf_dir)
    results = []
    for root in roots:
        results.extend(await asyncio.to_thread(app.state.indexer.index_directory, root, force))
    return {
        "total": len(results),
        "indexed": sum(item.status == "indexed" for item in results),
        "needs_ocr": sum(item.status == "needs_ocr" for item in results),
        "unchanged": sum(item.status == "unchanged" for item in results),
        "failed": [asdict(item) for item in results if item.status == "failed"],
    }


@app.delete("/api/documents/{document_id}", dependencies=[Depends(require_admin)])
async def delete_document(document_id: int, delete_file: bool = False) -> dict:
    documents = {item["id"]: item for item in app.state.database.list_documents()}
    document = documents.get(document_id)
    if not document or not app.state.database.delete_document(document_id):
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    if delete_file:
        file_path = Path(document["file_path"]).resolve()
        roots = (
            app.state.settings.upload_dir.resolve(),
            app.state.settings.law_pdf_dir.resolve(),
            app.state.settings.external_data_dir.resolve(),
        )
        if any(file_path == root or root in file_path.parents for root in roots) and file_path.exists():
            file_path.unlink()
    return {"deleted": True}


@app.get("/files/{document_id}")
async def document_file(document_id: int) -> FileResponse:
    documents = {item["id"]: item for item in app.state.database.list_documents()}
    document = documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    path = Path(document["file_path"]).resolve()
    roots = (
        app.state.settings.upload_dir.resolve(),
        app.state.settings.law_pdf_dir.resolve(),
        app.state.settings.external_data_dir.resolve(),
    )
    if not any(path == root or root in path.parents for root in roots) or not path.exists():
        raise HTTPException(status_code=404, detail="원본 파일을 찾을 수 없습니다.")
    media_type = guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=document["filename"])
