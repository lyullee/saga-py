from __future__ import annotations

"""Digital-twin → HAZOP → RAG safety decision support.

The Java digital-twin service already emits readings using the HAZOP table
vocabulary (``scenarioId``, ``tagId``, ``thresholdType`` and so on).  This
module keeps that contract at the API boundary, performs deterministic rule
evaluation first, then asks the RAG/LLM layer only to explain the result.  It
never invents a numeric limit when the local HAZOP table or indexed source
does not contain one.
"""

import asyncio
import json
import math
import re
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

from .config import Settings
from .database import Database
from .schemas import (
    DigitalTwinHazopRequest,
    DigitalTwinTagReading,
    HazopHit,
    HazopReference,
    HazopSop,
    HazopThresholdProposal,
    HazopThresholdProposalRequest,
    HazopThresholdProposalResponse,
    DigitalTwinHazopResponse,
)
from .text import expand_with_synonyms, normalize_text


# These are candidate sources for gaps.  They are deliberately marked
# ``catalog_only`` until a current edition is downloaded and indexed.  The
# response must never present this catalogue as proof of a legal requirement.
HAZOP_REFERENCE_CATALOG: tuple[dict[str, str], ...] = (
    {
        "label": "국가법령정보센터 고압가스 안전관리법·수소법·산업안전보건법",
        "source_type": "domestic_law_candidate",
        "source_url": "https://www.law.go.kr/",
    },
    {
        "label": "한국가스안전공사 KGS Code·상세기준",
        "source_type": "domestic_safety_code_candidate",
        "source_url": "https://cyber.kgs.or.kr/",
    },
    {
        "label": "KOSHA Guide 공정안전·위험성평가 지침",
        "source_type": "domestic_guidance_candidate",
        "source_url": "https://www.kosha.or.kr/",
    },
    {
        "label": "IEC 61882 HAZOP studies",
        "source_type": "international_methodology",
        "source_url": "https://webstore.iec.ch/en/publication/6047",
    },
    {
        "label": "ISO 19880-1 Gaseous hydrogen fuelling stations",
        "source_type": "international_standard",
        "source_url": "https://www.iso.org/standard/71940.html",
    },
    {
        "label": "NFPA 2 Hydrogen Technologies Code",
        "source_type": "international_code",
        "source_url": "https://www.nfpa.org/codes-and-standards/nfpa-2-standard-development/2",
    },
    {
        "label": "API RP 521 Pressure-relieving and Depressuring Systems",
        "source_type": "international_practice",
        "source_url": "https://www.api.org/products-and-services/standards/important-standards-announcements/rp-521",
    },
    {
        "label": "CCPS Guidelines for Hazard Evaluation Procedures",
        "source_type": "professional_guidance",
        "source_url": "https://www.aiche.org/ccps/resources/publications",
    },
)

BAD_QUALITY = {"BAD", "INVALID", "UNCERTAIN", "STALE", "ERROR", "나쁨", "불확실", "오류"}
NUMERIC_COMPARE_DIRS = {">", ">=", "<", "<=", "이상", "초과", "이하", "미만"}
SCENARIO_RE = re.compile(r"[^a-z0-9가-힣]+", re.IGNORECASE)


def _norm(value: Any) -> str:
    return normalize_text(str(value or "")).lower()


def _units_equivalent(left: Any, right: Any) -> bool:
    """Treat notation-only unit variants as equal, never convert dimensions."""
    a = _norm(left).replace(" ", "")
    b = _norm(right).replace(" ", "")
    if a == b:
        return True
    aliases = {
        "mpa_abs": "mpa", "mpaabsolute": "mpa", "mpa_g": "mpa", "mpag": "mpa",
        "deg_c": "degc", "°c": "degc", "c": "degc",
        "vol%h2": "vol%_h2", "vol%_hydrogen": "vol%_h2",
    }
    return aliases.get(a, a) == aliases.get(b, b)


def _scenario_key(value: str) -> str:
    value = _norm(value)
    return SCENARIO_RE.sub("-", value).strip("-") or "*"


def _parse_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _compare(value: float, threshold: float | None, direction: str) -> bool:
    if threshold is None or not math.isfinite(value):
        return False
    token = _norm(direction).replace(" ", "")
    if token in {">=", "이상"}:
        return value >= threshold
    if token in {">", "초과"}:
        return value > threshold
    if token in {"<=", "이하"}:
        return value <= threshold
    if token in {"<", "미만"}:
        return value < threshold
    # A label such as High/High-High is not a monitoring criterion.
    return False


def _effective_threshold(rule: dict[str, Any], reading: DigitalTwinTagReading) -> tuple[float | None, str | None]:
    threshold_type = _norm(rule.get("threshold_type") or "STATIC").replace("-", "_")
    raw = rule.get("threshold_value")
    try:
        threshold = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        threshold = None
    if threshold_type in {"", "static"}:
        return threshold, None
    if threshold_type in {"delta_start", "deltastart", "change_from_start", "변화_시작"}:
        if reading.baseline_start is None or threshold is None:
            return None, "시작 기준값 또는 변화 임계값이 없어 평가할 수 없습니다."
        # Match the Java evaluator exactly: DELTA thresholds are signed
        # increments, e.g. baseline + (-2) for a low-level deviation.
        return reading.baseline_start + threshold, None
    if threshold_type in {"delta_prev", "deltaprev", "change_from_previous", "변화_직전"}:
        if reading.baseline_prev is None or threshold is None:
            return None, "직전 기준값 또는 변화 임계값이 없어 평가할 수 없습니다."
        return reading.baseline_prev + threshold, None
    if threshold_type in {"none", "없음", "manual"}:
        return None, "정량 임계값이 없는 수동 판단 규칙입니다."
    return None, f"지원하지 않는 threshold_type({rule.get('threshold_type')})입니다."


def _direction_group(value: str) -> str:
    token = _norm(value).replace(" ", "")
    if token in {">", ">=", "이상", "초과"}:
        return "HIGH"
    if token in {"<", "<=", "이하", "미만"}:
        return "LOW"
    return token or "UNKNOWN"


def _threshold_readiness(rule: dict[str, Any]) -> tuple[list[str], bool]:
    """Return evidence gaps and whether the row cannot be evaluated at all."""
    gaps: list[str] = []
    threshold_type = _norm(rule.get("threshold_type") or "STATIC").replace("-", "_")
    if threshold_type not in {"static", "delta_start", "deltastart", "change_from_start", "변화_시작", "delta_prev", "deltaprev", "change_from_previous", "변화_직전"}:
        return [f"threshold_type={rule.get('threshold_type') or '없음'}은 수치 모니터링용이 아닙니다."], True
    try:
        threshold = float(rule.get("threshold_value"))
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isfinite(threshold):
        gaps.append("threshold_value 숫자값이 없습니다.")
    direction = _norm(rule.get("compare_dir") or "").replace(" ", "")
    if direction not in NUMERIC_COMPARE_DIRS:
        return [*gaps, "compare_dir는 이상/초과/이하/미만 또는 부등호 수치식이어야 합니다(High/High-High 사용 금지)."], True
    if not str(rule.get("unit") or "").strip():
        gaps.append("threshold 단위(unit)가 기재되지 않았습니다.")
    if not (str(rule.get("threshold_basis") or "").strip() or str(rule.get("threshold_source") or "").strip() or str(rule.get("standard_ref") or "").strip()):
        gaps.append("임계값을 설정한 근거(threshold_basis/threshold_source/standard_ref)가 없습니다.")
    confidence = _norm(rule.get("threshold_confidence") or "unverified")
    if confidence not in {"verified", "derived"}:
        gaps.append(f"임계값 근거 검토상태가 {rule.get('threshold_confidence') or 'unverified'}입니다.")
    return gaps, bool(not math.isfinite(threshold) or direction not in NUMERIC_COMPARE_DIRS)


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_rank(value: Any, severity: Any = "") -> int:
    parsed = _as_number(value)
    if parsed is not None:
        return max(0, min(9, int(parsed)))
    token = _norm(severity)
    return {"긴급": 4, "critical": 4, "경보": 3, "warning": 3, "주의": 2,
            "advisory": 2, "정보": 1}.get(token, 1)


def _normalize_external_rule(row: dict[str, Any], scenario_id: str) -> dict[str, Any] | None:
    """Normalize simulator catalogue rows to the SAGA HAZOP vocabulary.

    The digital-twin catalogue intentionally uses Korean column names and the
    Java service uses camelCase.  Keeping this adapter at the boundary avoids
    duplicating or importing the whole catalogue into SAGA's database.
    """
    if not isinstance(row, dict):
        return None
    tag_id = (row.get("tag_id") or row.get("tagId") or row.get("sensor_id")
              or row.get("sensorId") or row.get("센서_ID"))
    if not str(tag_id or "").strip():
        return None
    threshold = row.get("threshold_value")
    if threshold is None:
        threshold = row.get("threshold", row.get("임계값"))
    operator = (row.get("compare_dir") or row.get("compareDir") or row.get("operator")
                or row.get("연산자") or row.get("thresholdType") or "")
    severity = row.get("severity") or row.get("등급") or row.get("level") or "주의"
    source = (row.get("source") or row.get("source_id") or row.get("sourceId")
              or row.get("근거") or "digital-twin-hazop")
    basis = (row.get("threshold_basis") or row.get("thresholdBasis")
             or row.get("basis") or row.get("설정근거") or source)
    rule_id = row.get("id", row.get("rule_id", row.get("ruleId")))
    numeric_id = int(rule_id) if isinstance(rule_id, int) or (isinstance(rule_id, str) and rule_id.isdigit()) else None
    numeric_no = row.get("no")
    if numeric_no is not None:
        try:
            numeric_no = int(numeric_no)
        except (TypeError, ValueError):
            numeric_no = None
    return {
        "id": numeric_id,
        "no": numeric_no,
        "scenario_id": str(row.get("scenario_id") or row.get("scenarioId") or scenario_id or "*"),
        "scenario_name": str(row.get("scenario_name") or row.get("scenarioName")
                              or row.get("name") or row.get("시나리오명") or ""),
        "tag_id": str(tag_id),
        "item_name": str(row.get("item_name") or row.get("itemName") or row.get("variable")
                          or row.get("설비_라인") or row.get("node_id") or ""),
        "unit": str(row.get("unit") or row.get("단위") or ""),
        "guide_word": str(row.get("guide_word") or row.get("guideWord") or ""),
        "cond_text": str(row.get("cond_text") or row.get("condText") or ""),
        "threshold_type": str(row.get("threshold_type") or row.get("thresholdType") or "STATIC"),
        "threshold_value": threshold,
        "threshold_basis": str(basis or ""),
        "threshold_source": str(source or ""),
        # A catalogue row is accepted as the source-of-truth for this snapshot;
        # this is not a claim that the numeric limit is legally verified.
        "threshold_confidence": str(row.get("threshold_confidence") or row.get("thresholdConfidence") or "derived"),
        "compare_dir": str(operator),
        "severity": str(severity),
        "severity_rank": _as_rank(row.get("severity_rank") or row.get("severityRank"), severity),
        "risk_scenario": str(row.get("risk_scenario") or row.get("riskScenario")
                              or row.get("사고_전개조건") or row.get("시나리오명") or ""),
        "consequence": str(row.get("consequence") or row.get("결과") or ""),
        "emergency_action": str(row.get("emergency_action") or row.get("emergencyAction")
                                or row.get("비상조치") or ""),
        "future_measure": str(row.get("future_measure") or row.get("futureMeasure") or ""),
        "standard_ref": str(row.get("standard_ref") or row.get("standardRef") or source or ""),
        "source": str(source),
        "source_url": row.get("source_url") or row.get("sourceUrl"),
    }


class HazopEngine:
    """Evaluate one digital-twin snapshot and produce an SOP-style answer."""

    def __init__(self, database: Database, settings: Settings, reasoner: Any, pipeline: Any = None):
        self.database = database
        self.settings = settings
        self.reasoner = reasoner
        self.pipeline = pipeline

    def monitoring_readiness(self, scenario_id: str | None = None) -> dict[str, Any]:
        """Audit whether every active HAZOP row is fit for numeric monitoring."""
        rows = self.database.list_hazop_rules(scenario_id, None, False)
        issues: list[dict[str, Any]] = []
        ready_rows = 0
        for row in rows:
            gaps, blocking = _threshold_readiness(row)
            if gaps:
                issues.append({
                    "rule_id": row.get("id"),
                    "no": row.get("no"),
                    "scenario_id": row.get("scenario_id"),
                    "tag_id": row.get("tag_id"),
                    "blocking": blocking,
                    "gaps": gaps,
                })
            else:
                ready_rows += 1
        return {
            "scenario_id": scenario_id,
            "total_rules": len(rows),
            "ready_rules": ready_rows,
            "not_ready_rules": len(issues),
            "monitoring_ready": bool(rows) and not issues,
            "issues": issues,
        }

    async def propose_threshold(
        self, request: HazopThresholdProposalRequest
    ) -> HazopThresholdProposalResponse:
        """Propose numeric limits from indexed evidence without auto-activating them.

        A proposal is accepted only when the model quotes a passage that is
        present in a locally indexed document and the quoted passage contains
        the proposed number.  The returned rule remains ``pending`` and must
        be reviewed before importing it into the live HAZOP table.
        """
        query = expand_with_synonyms(
            " ".join(
                part for part in (
                    request.context,
                    request.scenario_id,
                    request.tag_id,
                    request.equipment_name,
                    request.variable,
                    request.unit,
                    "정상범위 경보 임계값 안전기준 수치 설정 근거",
                )
                if part
            )
        )
        try:
            rows = await asyncio.to_thread(
                self.database.hybrid_search,
                query,
                min(max(getattr(self.settings, "context_limit", 12), 8), 16),
                None,
                "CROSS",
            )
        except Exception:
            rows = []
        references = self._references(rows)
        indexed = [reference for reference in references if reference.status == "indexed"]
        limitations = [
            "이 결과는 임계값 후보이며 자동으로 HAZOP 모니터링 기준에 반영되지 않습니다. 안전관리자·설계·계측 담당자의 승인이 필요합니다.",
            "법적 의무·설계값·알람 설정값은 최신 원문과 설비 설계자료를 함께 확인해야 합니다.",
        ]
        if not indexed:
            return HazopThresholdProposalResponse(
                scenario_id=request.scenario_id,
                tag_id=request.tag_id,
                ready=False,
                candidates=[],
                references=self._catalog_references(),
                limitations=[
                    *limitations,
                    "로컬 색인에서 임계값을 직접 뒷받침하는 원문을 찾지 못했습니다. 수치를 추정하지 않았습니다.",
                ],
            )
        if not getattr(self.settings, "service_hub_api_key", "") or not self.reasoner:
            return HazopThresholdProposalResponse(
                scenario_id=request.scenario_id,
                tag_id=request.tag_id,
                ready=False,
                references=indexed,
                limitations=[*limitations, "Service Hub가 설정되지 않아 근거 인용형 후보 분석을 실행하지 못했습니다."],
            )
        evidence = "\n".join(
            f"[{ref.doc_code} p.{ref.page}] {ref.excerpt}" for ref in indexed
        )
        prompt = f"""다음 색인 원문만 사용하여 디지털 트윈 HAZOP 모니터링 임계값 후보를 제안하세요.

시나리오: {request.scenario_id}
태그: {request.tag_id}
설비: {request.equipment_name}
변수: {request.variable}
요청 단위: {request.unit}
추가 맥락: {request.context}

원문 근거:
{evidence}

규칙:
- 원문에 숫자와 비교 방향이 명시된 경우에만 후보를 만드세요. 원문에 없으면 candidates를 빈 배열로 두세요.
- threshold_type은 STATIC/DELTA_START/DELTA_PREV 중 하나, compare_dir는 이상/초과/이하/미만 또는 부등호만 사용하세요. High, High-High, Low 같은 guide word는 절대 사용하지 마세요.
- 후보 하나마다 doc_code, page, evidence_quote를 반드시 적고, evidence_quote는 위 원문에서 연속된 문장을 그대로 복사하세요. 인용문 안에 threshold_value 숫자가 실제로 있어야 합니다.
- 법령의 적용범위나 일반 설명을 설비 알람 수치로 추론하지 마세요. 서로 다른 단위의 값을 변환하지 마세요.
- JSON 객체 하나만 반환하세요. 스키마의 decision_notes에는 후보가 없는 이유나 추가로 필요한 설계자료를 적으세요.
"""
        try:
            proposal = await self.reasoner.structured_model(
                prompt,
                HazopThresholdProposal,
                "hazop_threshold_proposal",
                request.model or getattr(self.settings, "service_hub_model", "gpt-oss-20b"),
                getattr(self.settings, "reasoning_effort", "low"),
                4500,
            )
        except Exception as exc:
            return HazopThresholdProposalResponse(
                scenario_id=request.scenario_id,
                tag_id=request.tag_id,
                ready=False,
                references=indexed,
                limitations=[*limitations, f"근거 인용형 후보 분석에 실패했습니다: {type(exc).__name__}"],
            )
        valid: list[Any] = []
        rejected = 0
        for candidate in proposal.candidates:
            ref = next(
                (
                    item for item in indexed
                    if item.doc_code == candidate.doc_code and item.page == candidate.page
                ),
                None,
            )
            quote = normalize_text(candidate.evidence_quote)
            if ref is None or not quote or quote not in normalize_text(ref.excerpt):
                rejected += 1
                continue
            numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", quote)
            try:
                number_match = any(abs(float(value.replace(",", "")) - candidate.threshold_value) < 1e-9 for value in numbers)
            except ValueError:
                number_match = False
            if not number_match:
                rejected += 1
                continue
            if request.unit and candidate.unit and _norm(request.unit) != _norm(candidate.unit):
                rejected += 1
                continue
            valid.append(
                candidate.model_copy(
                    update={
                        "threshold_confidence": "pending",
                        "threshold_source": candidate.threshold_source or f"{candidate.doc_code} p.{candidate.page}",
                        "threshold_basis": candidate.threshold_basis or candidate.rationale or quote,
                    }
                )
            )
        if rejected:
            limitations.append(f"원문 인용·수치 검증을 통과하지 못한 후보 {rejected}건은 제외했습니다.")
        if proposal.decision_notes:
            limitations.append(f"분석 메모: {proposal.decision_notes}")
        return HazopThresholdProposalResponse(
            scenario_id=request.scenario_id,
            tag_id=request.tag_id,
            ready=bool(valid),
            candidates=valid,
            references=indexed,
            limitations=limitations,
        )

    @staticmethod
    def _equipment_for(reading: DigitalTwinTagReading, equipment: list[Any]) -> Any | None:
        if reading.equipment_id:
            for item in equipment:
                if item.equipment_id == reading.equipment_id:
                    return item
        for item in equipment:
            if reading.tag_id in item.tag_ids:
                return item
        return None

    def _references(self, rows: list[dict[str, Any]]) -> list[HazopReference]:
        refs: list[HazopReference] = []
        seen: set[tuple[str, int | None]] = set()
        for row in rows:
            key = (str(row.get("doc_code") or row.get("title") or ""), row.get("page"))
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                HazopReference(
                    label=f"{row.get('doc_code', '')} · {row.get('title', '')}".strip(" ·"),
                    source_type=str(row.get("doc_type") or "indexed"),
                    status="indexed",
                    doc_code=str(row.get("doc_code") or ""),
                    title=str(row.get("title") or ""),
                    page=int(row["page"]) if row.get("page") is not None else None,
                    excerpt=normalize_text(str(row.get("content") or ""))[:700],
                    source_url=str(row.get("source_url") or "") or None,
                )
            )
        return refs

    def _catalog_references(self) -> list[HazopReference]:
        return [
            HazopReference(
                label=item["label"],
                source_type=item["source_type"],
                status="catalog_only",
                source_url=item.get("source_url"),
                excerpt="현재 로컬 색인에서 원문을 확인하지 못한 보완 후보입니다. 정확한 조항·수치는 최신판을 확인해야 합니다.",
            )
            for item in HAZOP_REFERENCE_CATALOG
        ]

    @staticmethod
    def _deterministic_answer(
        request: DigitalTwinHazopRequest,
        hits: list[HazopHit],
        references: list[HazopReference],
        missing: list[str],
        unevaluated: list[str],
        threshold_gaps: list[str] | None = None,
    ) -> str:
        lines: list[str] = []
        if hits:
            lines.append("## 판단 요약\n")
            lines.append(
                f"현재 디지털 트윈 스냅샷에서 **{len(hits)}개 HAZOP 조건이 감지**되었습니다. "
                "아래 조치는 해당 HAZOP 행의 비상조치와 결과를 우선 정리한 것입니다."
            )
            lines.append("\n## 즉시 조치(현장 SOP 순서)\n")
            for index, hit in enumerate(hits, start=1):
                label = hit.item_name or hit.tag_id
                lines.append(f"{index}. **{label} ({hit.tag_id})** — {hit.severity} 조건")
                if hit.emergency_action:
                    lines.append(f"   - HAZOP 비상조치: {hit.emergency_action}")
                else:
                    lines.append("   - HAZOP 행에 명시된 비상조치가 없어 현장 비상대응계획과 책임자 지시에 따라 안전한 격리·대피 여부를 판단합니다.")
                if hit.risk_scenario:
                    lines.append(f"   - 예상 시나리오: {hit.risk_scenario}")
                if hit.consequence:
                    lines.append(f"   - 가능한 결과: {hit.consequence}")
                if hit.standard_ref:
                    lines.append(f"   - 연결 기준: {hit.standard_ref}")
            lines.append("\n## 복구·재발 방지\n")
            measures = [hit.future_measure for hit in hits if hit.future_measure]
            lines.extend(f"- {measure}" for measure in measures)
            if not measures:
                lines.append("- 원인 확인과 재가동 승인은 현장 절차·작업허가·안전관리자 확인 후 진행합니다.")
            lines.append("\n## 통제·격리 원칙\n")
            lines.append("- 누출·화재·급격한 압력 변화가 의심되면 위험구역 접근을 제한하고, 설계된 ESD·인터록과 현장 비상대응계획에 따라 대피·격리 여부를 판단합니다.")
            lines.append("- 이 API는 밸브·ESD를 자동 조작하지 않으며, 권한 있는 운전자가 승인된 절차로만 조작합니다.")
            lines.append("\n## 확인·기록 및 재가동 승인\n")
            lines.append("- 태그 값·단위·quality·시각, 알람/event 로그, 조치자·승인자, HAZOP 행 번호와 원인조사 결과를 남깁니다.")
            lines.append("- 원인 제거, 계기·인터록 정상 확인, 필요한 정비·시험과 작업허가 종료를 확인한 뒤 책임자가 재가동을 승인합니다.")
        else:
            lines.append("## 판단 요약\n")
            lines.append("현재 입력값에서 HAZOP 임계조건이 확인되지 않았습니다. 다만 모든 태그가 완전히 평가된 것은 아닐 수 있으므로 아래 제한사항을 확인해 주세요.")
            lines.append("\n## 정상 감시 원칙\n")
            lines.append("- 정상 판정은 전달된 태그와 현재 HAZOP 규칙 범위에 한정됩니다. 다음 측정 주기와 알람 상태를 계속 감시합니다.")
            lines.append("- 미등록·품질 불량·오래된 값이 있으면 정상으로 확정하지 말고 운전 책임자에게 확인을 요청합니다.")
        if missing:
            lines.append("\n## HAZOP 미등록 항목\n")
            lines.append("다음 태그는 현재 HAZOP 테이블에 직접 대응하는 행이 없습니다: " + ", ".join(f"`{tag}`" for tag in missing) + ".")
        if unevaluated:
            lines.append("\n## 추가 확인이 필요한 입력\n")
            lines.extend(f"- {item}" for item in unevaluated)
        if threshold_gaps:
            lines.append("\n## 수치 임계값·근거 검토 필요\n")
            lines.append("아래 행은 감시 기준으로 사용하기 전에 수치, 단위, 비교방향, 근거 문서를 확인해야 합니다.")
            lines.extend(f"- {item}" for item in list(dict.fromkeys(threshold_gaps)))
        if references:
            indexed = [ref for ref in references if ref.status == "indexed"]
            if indexed:
                lines.append("\n## 색인 근거를 반영한 보완 설명\n")
                lines.append("HAZOP 행으로 직접 판단하기 어려운 부분은 아래 색인 문서의 관련 문단을 참고해 보완해야 합니다. 원문에 없는 수치나 의무는 추정하지 않았습니다.")
                for ref in indexed[:6]:
                    lines.append(f"- **{ref.label}** (p.{ref.page or '-'}): {ref.excerpt}")
        return "\n".join(lines).strip()

    @staticmethod
    def _structured_sections(
        request: DigitalTwinHazopRequest,
        status: str,
        hits: list[HazopHit],
        worst_rank: int,
        stale_tags: list[str],
        missing_rule_tags: list[str],
        data_quality: list[str],
        threshold_gaps: list[str],
    ) -> dict[str, Any]:
        """Build machine-readable SOP gates independently from the LLM prose.

        These sections are intentionally conservative.  They tell an operator
        what must be checked and who must authorize the next step, rather than
        issuing an automatic valve/ESD command or inventing a setpoint.
        """
        severity_text = " ".join(_norm(hit.severity) for hit in hits)
        priority = "normal"
        if status == "PARTIAL":
            priority = "attention"
        if hits:
            # The Java HAZOP table defines 주의=1, 경보=2, 긴급=3.
            priority = "emergency" if worst_rank >= 3 or any(word in severity_text for word in ("비상", "긴급", "critical")) else "warning"

        immediate: list[str] = []
        for hit in hits:
            label = hit.item_name or hit.tag_id
            if hit.emergency_action:
                immediate.append(f"{label}({hit.tag_id}) HAZOP 조치: {hit.emergency_action}")
            else:
                immediate.append(f"{label}({hit.tag_id})은 HAZOP 행에 비상조치가 없으므로 현장 비상대응계획과 책임자 지시에 따라 조치 범위를 결정합니다.")
        if not immediate and status == "PARTIAL":
            immediate.append("판정에 필요한 태그·기준값·단위·품질 정보를 보완하기 전까지 정상 상태로 확정하지 않습니다.")
        if not immediate and status == "NORMAL":
            immediate.append("현재 입력된 태그는 HAZOP 임계조건을 넘지 않았습니다. 다음 측정 주기와 알람 상태를 계속 감시합니다.")

        isolation: list[str] = []
        if hits:
            isolation.extend([
                "누출·화재·급격한 압력 변화가 의심되면 사람을 위험구역으로 보내지 말고 현장 비상대응계획의 대피·통제·신고 절차를 우선합니다.",
                "격리·감압·ESD 실행은 설계된 자동 기능과 권한 있는 운전자가 승인된 절차에 따라 수행하며, API가 임의 조작을 지시하지 않습니다.",
            ])
        elif status == "PARTIAL":
            isolation.append("데이터 신뢰성이 확보될 때까지 재가동·우회운전을 단정하지 말고 운전 책임자에게 상태를 보고합니다.")

        verification = [
            "태그 값, 단위, quality, 타임스탬프와 알람 발생 시각을 원본 historian/SCADA 기록과 대조합니다.",
            "가능한 경우 위험구역에 진입하지 않는 독립 계기·원격 신호·인터록 상태로 측정값을 교차 확인합니다.",
        ]
        verification.extend(
            f"{hit.tag_id}의 현재값({hit.value}{hit.unit})과 HAZOP 조건({hit.compare_dir} {hit.effective_threshold})의 비교 결과를 기록합니다."
            for hit in hits
        )
        if stale_tags or data_quality:
            verification.append("품질 불량·신선도 초과·단위 불일치 태그는 센서/통신 상태를 확인하고 보정 전 재판정하지 않습니다.")
        if missing_rule_tags:
            verification.append("미등록 태그는 최신 HAZOP·국내 법령·KGS Code 원문을 검토해 담당자가 규칙을 승인한 후 테이블에 등록합니다.")
        if threshold_gaps:
            verification.append("수치 임계값·단위·비교방향·설정근거가 확인되지 않은 HAZOP 행은 자동 경보 기준으로 승인하지 않습니다.")

        restart = [
            "원인과 영향 범위를 확인하고 필요한 정비·교정·누출/압력 시험 결과를 기록합니다.",
            "HAZOP 조건이 해소되고 알람·ESD·인터록이 정상 복귀했는지 권한 있는 담당자가 확인합니다.",
            "작업허가, 위험성평가, 현장 순회 확인과 책임자 승인 없이는 재가동하지 않습니다.",
        ] if hits or status == "PARTIAL" else [
            "정상 감시를 유지하고 다음 운전 주기에서 동일 조건이 재발하지 않는지 확인합니다."
        ]
        records = [
            "station_id, scenario_id, 평가시각, 모델 버전과 평가 결과(status/state_key)",
            "관련 태그의 값·단위·quality·timestamp·기준값(baseline) 및 알람/event 로그",
            "운전자 조치, ESD/인터록 상태, 대피·신고·작업허가·정비 기록과 승인자",
            "적용된 HAZOP 행 번호·standard_ref와 RAG 문서 코드·페이지",
        ]
        escalation = [
            "비상·누출·화재·부상 가능성이 있으면 현장 비상연락망과 관계기관 신고 절차를 따릅니다.",
            "반복 경보, 원인 불명, 센서 간 불일치 또는 HAZOP 미등록 상태는 안전관리자·운전 책임자에게 즉시 에스컬레이션합니다.",
        ] if hits or status == "PARTIAL" else [
            "이상 징후가 새로 발생하면 즉시 운전 책임자와 안전관리자에게 공유합니다."
        ]
        return {
            "priority": priority,
            "immediate_actions": list(dict.fromkeys(immediate)),
            "isolation_evacuation": list(dict.fromkeys(isolation)),
            "verification_steps": list(dict.fromkeys(verification)),
            "restart_requirements": list(dict.fromkeys(restart)),
            "records_to_capture": records,
            "escalation": list(dict.fromkeys(escalation)),
        }

    def _prompt(
        self,
        request: DigitalTwinHazopRequest,
        hits: list[HazopHit],
        references: list[HazopReference],
        fallback: str,
        stale_tags: list[str] | None = None,
        missing_rule_tags: list[str] | None = None,
        unevaluated: list[str] | None = None,
        data_quality: list[str] | None = None,
        threshold_gaps: list[str] | None = None,
    ) -> str:
        evidence = [
            {
                "tag": hit.tag_id,
                "item": hit.item_name,
                "value": hit.value,
                "unit": hit.unit,
                "severity": hit.severity,
                "threshold": hit.effective_threshold,
                "direction": hit.compare_dir,
                "threshold_basis": hit.threshold_basis,
                "threshold_source": hit.threshold_source,
                "threshold_confidence": hit.threshold_confidence,
                "risk": hit.risk_scenario,
                "consequence": hit.consequence,
                "emergency_action": hit.emergency_action,
                "future_measure": hit.future_measure,
                "standard_ref": hit.standard_ref,
            }
            for hit in hits
        ]
        indexed = [
            {"code": ref.doc_code, "title": ref.title, "page": ref.page, "excerpt": ref.excerpt}
            for ref in references
            if ref.status == "indexed"
        ]
        return f"""당신은 수소충전소 디지털 트윈의 안전관리자입니다. 다음 상태를 HAZOP 기반 SOP로 설명하세요.

질문/상태 설명: {request.context_text or request.condition or '디지털 트윈 상태 평가'}
시나리오: {request.scenario_id}
HAZOP 감지행(JSON): {json.dumps(evidence, ensure_ascii=False)}
색인 문서 근거(JSON): {json.dumps(indexed, ensure_ascii=False)}
평가 제한(JSON): {json.dumps({"stale_tags": stale_tags or [], "missing_rule_tags": missing_rule_tags or [], "unevaluated": unevaluated or [], "data_quality": data_quality or [], "threshold_gaps": threshold_gaps or []}, ensure_ascii=False)}

반드시 지킬 규칙:
1) HAZOP 행의 감지 사실과 비상조치를 최우선으로 해석하고, 다음 제목을 포함한 충분히 상세한 한국어로 작성하세요: '판단 요약', '즉시 조치', '통제·격리·대피 원칙', '원인/결과', '확인·기록', '복구·재가동 승인', '재발 방지'.
2) 누출·화재·과압 가능성이 있으면 사람을 위험 구역으로 보내거나 임의로 밸브를 조작하라고 하지 말고, 설계된 ESD/인터록과 현장 비상대응계획·대피·신고 절차를 우선하도록 쓰세요.
3) 색인 문서가 직접 뒷받침하지 않는 법적 의무·수치·거리·압력은 만들지 마세요. 색인 근거가 없으면 '일반적인 안전 설명'이라고 명시하세요.
4) 검색 결과 목록을 그대로 나열하지 말고, 운영자가 왜 그런 조치를 해야 하는지 쉽게 풀어 설명하세요. HAZOP 행에 없는 내용은 별도 '보완 설명'으로 구분하세요.
5) 즉시조치와 권고사항을 구분하고, '반드시/의무'라는 표현은 원문 근거가 있을 때만 사용하세요. 원문에 없는 압력·거리·시간·밸브 번호·대피 반경은 만들지 마세요.
6) 센서값이 오래됐거나 quality가 불량하면 정상으로 단정하지 말고 재확인·에스컬레이션 절차를 제시하세요. 재가동은 원인 제거, 인터록/ESD 확인, 작업허가와 책임자 승인을 통과해야 한다고 명시하세요.
7) 평가 제한에 있는 미등록·미평가 태그를 HAZOP 정상판정으로 표현하지 마세요.
8) 다음 초안은 참고만 하고 오류가 있으면 고치세요:
{fallback}
"""

    async def _supplement(self, request: DigitalTwinHazopRequest, hits: list[HazopHit], missing: list[str]) -> list[HazopReference]:
        query_parts = [request.context_text, request.condition, request.scenario_id]
        query_parts.extend(f"{hit.tag_id} {hit.item_name} {hit.risk_scenario}" for hit in hits)
        query_parts.extend(missing)
        query = expand_with_synonyms(" ".join(part for part in query_parts if part))
        if not query:
            return []
        try:
            rows = await asyncio.to_thread(
                self.database.hybrid_search,
                query,
                min(max(getattr(self.settings, "context_limit", 12), 8), 16),
                None,
                "CROSS",
            )
        except Exception:
            return []
        return self._references(rows)

    async def evaluate(
        self,
        request: DigitalTwinHazopRequest,
        *,
        direct: bool = False,
    ) -> DigitalTwinHazopResponse:
        """Evaluate a snapshot.

        ``direct=True`` is the real-time digital-twin path: it uses only the
        supplied HAZOP rows, performs deterministic numeric comparisons, and
        never invokes RAG or an LLM reviewer.  The regular endpoint preserves
        the richer RAG/LLM behaviour used by operator conversations.
        """
        started = perf_counter()
        evaluated_at = datetime.now(UTC)
        if request.hazop_rules:
            rules = [
                normalized
                for normalized in (_normalize_external_rule(row, request.scenario_id) for row in request.hazop_rules)
                if normalized is not None
            ]
        else:
            rules = await asyncio.to_thread(
                self.database.list_hazop_rules, request.scenario_id, None, False
            )
        rules_by_tag: dict[str, list[dict[str, Any]]] = {}
        for rule in rules:
            rules_by_tag.setdefault(str(rule["tag_id"]), []).append(rule)
        equipment_map = {item.equipment_id: item for item in request.equipment}
        hits: list[HazopHit] = []
        unevaluated: list[str] = []
        stale_tags: list[str] = []
        missing_rule_tags: list[str] = []
        data_quality: list[str] = []
        threshold_gaps: list[str] = []
        evaluated_count = 0
        for reading in request.readings:
            tag = reading.tag_id
            if reading.value is None:
                unevaluated.append(f"{tag}: 측정값이 없습니다.")
                continue
            try:
                numeric = float(reading.value)
            except (TypeError, ValueError):
                unevaluated.append(f"{tag}: 숫자형으로 해석할 수 없는 측정값입니다.")
                continue
            if not math.isfinite(numeric):
                unevaluated.append(f"{tag}: 유한한 숫자가 아닙니다.")
                continue
            quality = _norm(reading.quality).upper()
            if quality in BAD_QUALITY:
                data_quality.append(f"{tag}: quality={reading.quality}")
            timestamp = _parse_timestamp(reading.timestamp)
            if timestamp and evaluated_at - timestamp > timedelta(seconds=request.max_staleness_seconds):
                stale_tags.append(tag)
                data_quality.append(f"{tag}: 마지막 측정시각이 {request.max_staleness_seconds}초보다 오래되었습니다.")
            candidates = rules_by_tag.get(tag, [])
            if not candidates:
                missing_rule_tags.append(tag)
                continue
            matched = False
            equipment = equipment_map.get(reading.equipment_id or "") or self._equipment_for(reading, request.equipment)
            rule_evaluated = False
            for rule in candidates:
                rule_gaps, blocking_gap = _threshold_readiness(rule)
                threshold_gaps.extend(
                    f"{tag} No.{rule.get('no') or rule.get('id') or '-'}: {gap}"
                    for gap in rule_gaps
                )
                if blocking_gap:
                    unevaluated.append(
                        f"{tag}: 수치 임계값/비교방향이 없어 이 HAZOP 행은 모니터링 판정에서 제외했습니다."
                    )
                    continue
                rule_unit = _norm(rule.get("unit") or "")
                reading_unit = _norm(reading.unit or "")
                if rule_unit and reading_unit and not _units_equivalent(rule_unit, reading_unit):
                    data_quality.append(f"{tag}: 입력 단위({reading.unit})와 HAZOP 단위({rule.get('unit')})가 다릅니다.")
                    unevaluated.append(f"{tag}: 단위 변환 규칙이 없어 해당 HAZOP 행을 평가하지 않았습니다.")
                    continue
                threshold, reason = _effective_threshold(rule, reading)
                if reason:
                    unevaluated.append(f"{tag}: {reason}")
                    continue
                rule_evaluated = True
                if not _compare(numeric, threshold, str(rule.get("compare_dir") or "")):
                    continue
                matched = True
                hits.append(
                    HazopHit(
                        rule_id=int(rule["id"]) if rule.get("id") is not None else None,
                        no=int(rule["no"]) if rule.get("no") is not None else None,
                        scenario_id=str(rule.get("scenario_id") or request.scenario_id),
                        tag_id=tag,
                        equipment_id=reading.equipment_id or getattr(equipment, "equipment_id", None),
                        equipment_name=reading.equipment_name or getattr(equipment, "name", None),
                        item_name=str(rule.get("item_name") or ""),
                        value=numeric,
                        unit=reading.unit or str(rule.get("unit") or ""),
                        guide_word=str(rule.get("guide_word") or ""),
                        threshold_value=float(rule["threshold_value"]) if rule.get("threshold_value") is not None else None,
                        effective_threshold=threshold,
                        threshold_basis=str(rule.get("threshold_basis") or ""),
                        threshold_source=str(rule.get("threshold_source") or ""),
                        threshold_confidence=str(rule.get("threshold_confidence") or "unverified"),
                        compare_dir=str(rule.get("compare_dir") or ""),
                        severity=str(rule.get("severity") or "주의"),
                        severity_rank=int(rule.get("severity_rank") or 1),
                        risk_scenario=str(rule.get("risk_scenario") or ""),
                        consequence=str(rule.get("consequence") or ""),
                        emergency_action=str(rule.get("emergency_action") or ""),
                        future_measure=str(rule.get("future_measure") or ""),
                        standard_ref=str(rule.get("standard_ref") or ""),
                        source=str(rule.get("source") or "hazop_table"),
                        source_url=str(rule.get("source_url") or "") or None,
                    )
                )
            if candidates and rule_evaluated:
                evaluated_count += 1
            elif candidates and tag not in missing_rule_tags and not any(tag in item for item in unevaluated):
                # A rule row exists but could not be quantitatively evaluated.
                unevaluated.append(f"{tag}: HAZOP 행은 있으나 정량 비교가 불가능합니다.")

        # Keep the most severe matching row for identical tag/condition pairs;
        # Java's evaluator uses the same severity-rank ordering.
        unique: dict[tuple[str, str], HazopHit] = {}
        for hit in hits:
            key = (hit.tag_id, _direction_group(hit.compare_dir))
            previous = unique.get(key)
            if previous is None or hit.severity_rank > previous.severity_rank:
                unique[key] = hit
        hits = sorted(unique.values(), key=lambda item: (-item.severity_rank, item.tag_id, item.no or 0))
        worst = hits[0] if hits else None
        if not request.readings:
            status = "UNKNOWN"
        elif hits:
            status = "WARNING"
        elif missing_rule_tags or unevaluated or stale_tags or data_quality or threshold_gaps:
            status = "PARTIAL"
        else:
            status = "NORMAL"
        state_key = f"{_scenario_key(request.scenario_id)}|{status}"
        # Do not attach arbitrary KGS paragraphs to a normal snapshot.  RAG
        # supplementation is meaningful only for an alarm or an unregistered
        # tag (the operator can still request ordinary chat separately).
        refs = [] if direct else (
            await self._supplement(request, hits, missing_rule_tags)
            if (hits or missing_rule_tags) else []
        )
        if hits:
            table_refs = [
                HazopReference(
                    label=f"HAZOP 테이블 · {hit.tag_id} · {hit.item_name or '설비 상태'}",
                    source_type="HAZOP",
                    status="hazop_table",
                    doc_code=str(hit.scenario_id),
                    title=hit.risk_scenario or hit.item_name,
                    excerpt=(hit.emergency_action or hit.consequence or "감지된 HAZOP 조건")[:700],
                    source_url=hit.source_url,
                )
                for hit in hits
            ]
            refs = table_refs + refs
        monitoring_ready = bool(request.readings) and evaluated_count == len(request.readings) and not (
            missing_rule_tags or unevaluated or stale_tags or data_quality or threshold_gaps
        )
        limitations = [
            "이 결과는 디지털 트윈과 HAZOP/RAG를 이용한 의사결정 지원이며, 설계된 ESD·인터록·현장 비상대응계획과 안전관리자 판단이 우선합니다.",
        ]
        if direct:
            limitations.append(
                "직답 모드에서는 전달된 디지털 트윈 HAZOP 행과 센서값만 즉시 비교했습니다. "
                "LLM 추론·RAG 보완·법령 최신성 검토는 실행하지 않았습니다."
            )
        if missing_rule_tags:
            limitations.append("HAZOP 테이블에 없는 태그는 현재 표의 임계값으로 판정하지 않았습니다. 최신 국내·해외 기준 원문을 확인해 테이블을 보완해야 합니다.")
        if stale_tags or data_quality:
            limitations.append("측정 품질 또는 신선도가 확인되지 않은 값이 포함되어 결과를 정상 상태로 단정할 수 없습니다.")
        if threshold_gaps:
            limitations.append("수치 임계값 또는 설정 근거가 검토되지 않은 행이 있어 운영용 자동 경보 기준으로 승인할 수 없습니다.")
        if (hits or missing_rule_tags or unevaluated or stale_tags or data_quality or threshold_gaps) and not refs:
            limitations.append("로컬 색인에서 직접 확인한 보완 문서가 없습니다. 아래 후보 기준은 catalog_only이며 조항·수치의 근거로 사용하지 마세요.")
            refs = self._catalog_references()
        fallback = self._deterministic_answer(
            request, hits, refs, missing_rule_tags, unevaluated, threshold_gaps
        )
        answer = fallback
        has_indexed_refs = any(ref.status == "indexed" for ref in refs)
        source_status = "supplemented" if has_indexed_refs else (
            "hazop_grounded" if hits or status == "NORMAL" else "llm_general"
        )
        if (not direct and request.generate_sop and request.interpret
                and getattr(self.settings, "service_hub_api_key", "") and self.reasoner):
            try:
                answer = await self.reasoner.answer(
                    self._prompt(
                        request,
                        hits,
                        refs,
                        fallback,
                        stale_tags,
                        missing_rule_tags,
                        unevaluated,
                        data_quality,
                        threshold_gaps,
                    ),
                    model=request.model or getattr(self.settings, "service_hub_model", None),
                    reasoning_effort=getattr(self.settings, "reasoning_effort", "low"),
                    max_tokens={"concise": 2600, "standard": 4200, "detailed": 6500, "very_detailed": 9000}.get(getattr(self.settings, "answer_length", "standard"), 4200),
                ) or fallback
                source_status = "supplemented" if has_indexed_refs else (
                    "hazop_grounded" if hits or status == "NORMAL" else "llm_general"
                )
            except Exception:
                answer = fallback
        if not request.generate_sop:
            sop = None
        else:
            sections = self._structured_sections(
                request,
                status,
                hits,
                worst.severity_rank if worst else 0,
                stale_tags,
                missing_rule_tags,
                data_quality,
                threshold_gaps,
            )
            sop = HazopSop(
                answer=answer,
                source_status=source_status,
                limitations=limitations,
                references=refs,
                **sections,
            )
        response = DigitalTwinHazopResponse(
            station_id=request.station_id,
            scenario_id=request.scenario_id,
            evaluated_at=evaluated_at,
            status=status,
            worst_severity=worst.severity if worst else None,
            worst_rank=worst.severity_rank if worst else 0,
            hit_count=len(hits),
            hits=hits,
            unevaluated=list(dict.fromkeys(unevaluated)),
            stale_tags=list(dict.fromkeys(stale_tags)),
            missing_rule_tags=list(dict.fromkeys(missing_rule_tags)),
            evaluated_tag_count=evaluated_count,
            requested_tag_count=len(request.readings),
            state_key=state_key,
            sop=sop,
            data_quality=list(dict.fromkeys(data_quality)),
            monitoring_ready=monitoring_ready,
            threshold_gaps=list(dict.fromkeys(threshold_gaps)),
            processing_ms=max(0, int(round((perf_counter() - started) * 1000))),
        )
        try:
            evaluation_id = await asyncio.to_thread(
                self.database.save_hazop_evaluation,
                request.station_id,
                request.scenario_id,
                status,
                request.model_dump(mode="json"),
                response.model_dump(mode="json"),
            )
            response.evaluation_id = evaluation_id
        except Exception:
            # An evaluation response is still useful if an old read-only DB
            # cannot persist the audit row.
            pass
        return response

    async def evaluate_direct(self, request: DigitalTwinHazopRequest) -> DigitalTwinHazopResponse:
        """Low-latency deterministic HAZOP evaluation for digital-twin calls."""
        # Explicitly turn off interpretation in the copied request as a guard
        # against future code accidentally re-enabling the LLM branch.
        direct_request = request.model_copy(update={"interpret": False})
        response = await self.evaluate(direct_request, direct=True)
        response.impact_results = [row for row in request.impact_results
                                   if row.get("calculation_status") == "calculated"]
        return response
