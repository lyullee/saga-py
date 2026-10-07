from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)


AnswerLength = Literal["concise", "standard", "detailed", "very_detailed"]
ChatMode = Literal["rag", "chat"]
KnowledgeMode = Literal["standards", "operations", "incidents"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str | None = Field(default=None, max_length=80)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    model: str | None = Field(default=None, max_length=120)
    provider: Literal["service_hub", "groq"] = "service_hub"
    language: Literal["ko", "en"] = "ko"
    answer_length: AnswerLength | None = None
    # ``rag`` searches the local safety-document index. ``chat`` deliberately
    # skips retrieval so the user can have an ordinary LLM conversation.
    mode: ChatMode = "rag"
    # Which knowledge interface is active. ``standards`` preserves the
    # original KGS/law/rule RAG. The two external source modes are deliberately
    # separate so operational statistics and incident reports are not mixed
    # into normative compliance answers.
    knowledge_mode: KnowledgeMode = "standards"


class DigitalTwinDirectChatRequest(BaseModel):
    """One-pass LLM answer for an already evaluated digital-twin snapshot."""

    message: str = Field(min_length=1, max_length=10000)
    provider: Literal["service_hub", "groq"] = "service_hub"
    max_tokens: int = Field(default=1200, ge=128, le=3000)


class DigitalTwinMainAssistantRequest(BaseModel):
    """Isolated contract for the digital twin's main operations assistant."""

    question: str = Field(min_length=1, max_length=1200)
    context: dict[str, Any] = Field(default_factory=dict)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    request_kind: Literal["user_query", "automatic_analysis"] = "user_query"
    provider: Literal["service_hub", "groq"] = "service_hub"
    language: Literal["ko", "en"] = "ko"
    # Main-monitor answers are an operator headline. The digital twin renders
    # deterministic consequence cards and the full response plan separately.
    max_tokens: int = Field(default=900, ge=128, le=3000)


class DigitalTwinSensorAssistantRequest(BaseModel):
    """Isolated contract for one selected sensor and its local process context."""

    sensor_id: str = Field(min_length=1, max_length=40)
    question: str = Field(default="", max_length=1200)
    context: dict[str, Any] = Field(default_factory=dict)
    request_kind: Literal["user_query", "automatic_analysis"] = "automatic_analysis"
    provider: Literal["service_hub", "groq"] = "service_hub"
    language: Literal["ko", "en"] = "ko"
    max_tokens: int = Field(default=2200, ge=128, le=3000)


class Citation(BaseModel):
    number: int
    document_id: int
    chunk_id: int
    doc_type: str
    doc_code: str
    title: str
    hierarchy: str
    page: int
    filename: str
    excerpt: str
    score: float
    source_url: str | None = None


class CodeSuggestion(BaseModel):
    doc_code: str
    title: str
    doc_type: str


class ChatResponse(BaseModel):
    conversation_id: str
    log_id: int
    answer: str
    language: Literal["ko", "en"] = "ko"
    citations: list[Citation]
    rewritten_query: str
    intent: str
    suggestions: list[CodeSuggestion] = Field(default_factory=list)
    answer_mode: Literal["rag", "llm_only", "clarification"] = "rag"
    mode: ChatMode = "rag"
    knowledge_mode: KnowledgeMode = "standards"
    model: str = ""
    # The streaming UI keeps the first-pass answer visible while the second
    # pass validates and rewrites it.  ``answer`` remains the final answer for
    # existing API clients; these fields make the two stages explicit for
    # clients that want to render both.
    draft_answer: str = ""
    final_answer: str = ""
    review_applied: bool = False


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str
    intent: Literal["fact", "summary", "comparison", "procedure", "general"]
    domain: Literal["CODE", "RULE", "LAW", "CROSS", "GENERAL"]
    keywords: list[str]
    document_codes: list[str]
    retrieval_required: bool


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[int]


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    citation_number: int
    exact_quote: str


class GroundedClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[EvidenceClaim]
    insufficient_evidence: bool


class AnswerReview(BaseModel):
    """Second-pass quality gate for a completed answer."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    revised_answer: str = ""
    issues: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    id: int
    doc_type: str
    doc_code: str
    title: str
    filename: str
    page_count: int
    chunk_count: int
    indexed_at: str
    source_url: str | None = None


class FeedbackRequest(BaseModel):
    score: Literal[-1, 0, 1]


class LawSyncRequest(BaseModel):
    """Request for an explicit National Law Information Center sync."""

    keys: list[str] | None = Field(default=None, max_length=20)
    force: bool = False


class ExternalSourceSyncRequest(BaseModel):
    """Request for the public NREL/HIAD source importer."""

    sources: list[Literal["nrel", "hiad"]] = Field(
        default_factory=lambda: ["nrel", "hiad"], max_length=2
    )
    force: bool = False


class HazopRuleInput(BaseModel):
    """HAZOP rule row compatible with the existing Java ``hazop_rule`` table."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: int | None = None
    no: int | None = None
    scenario_id: str = Field(
        default="*", max_length=40,
        validation_alias=AliasChoices("scenario_id", "scenarioId"),
    )
    scenario_name: str = Field(
        default="", max_length=120,
        validation_alias=AliasChoices("scenario_name", "scenarioName"),
    )
    tag_id: str = Field(
        min_length=1, max_length=80,
        validation_alias=AliasChoices("tag_id", "tagId"),
    )
    item_name: str = Field(
        default="", max_length=200,
        validation_alias=AliasChoices("item_name", "itemName"),
    )
    unit: str = Field(default="", max_length=30)
    normal_range: str = Field(
        default="", max_length=160,
        validation_alias=AliasChoices("normal_range", "normalRange"),
    )
    relevance: str = Field(default="직접관련", max_length=30)
    guide_word: str = Field(
        default="", max_length=60,
        validation_alias=AliasChoices("guide_word", "guideWord"),
    )
    cond_text: str = Field(
        default="", max_length=160,
        validation_alias=AliasChoices("cond_text", "condText"),
    )
    threshold_type: str = Field(
        default="STATIC", max_length=30,
        validation_alias=AliasChoices("threshold_type", "thresholdType"),
    )
    threshold_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("threshold_value", "thresholdValue"),
    )
    threshold_basis: str = Field(
        default="",
        max_length=4000,
        validation_alias=AliasChoices("threshold_basis", "thresholdBasis"),
    )
    threshold_source: str = Field(
        default="",
        max_length=1000,
        validation_alias=AliasChoices("threshold_source", "thresholdSource"),
    )
    threshold_confidence: Literal["verified", "derived", "unverified", "pending"] = Field(
        default="unverified",
        validation_alias=AliasChoices("threshold_confidence", "thresholdConfidence"),
    )
    compare_dir: str = Field(
        default="", max_length=20,
        validation_alias=AliasChoices("compare_dir", "compareDir"),
    )
    severity: str = Field(default="주의", max_length=30)
    severity_rank: int = Field(
        default=1, ge=0, le=9,
        validation_alias=AliasChoices("severity_rank", "severityRank"),
    )
    risk_scenario: str = Field(
        default="", max_length=5000,
        validation_alias=AliasChoices("risk_scenario", "riskScenario"),
    )
    consequence: str = Field(default="", max_length=5000)
    emergency_action: str = Field(
        default="", max_length=10000,
        validation_alias=AliasChoices("emergency_action", "emergencyAction"),
    )
    future_measure: str = Field(
        default="", max_length=10000,
        validation_alias=AliasChoices("future_measure", "futureMeasure"),
    )
    standard_ref: str = Field(
        default="", max_length=3000,
        validation_alias=AliasChoices("standard_ref", "standardRef"),
    )
    source: str = Field(default="digital-twin-hazop", max_length=120)
    source_url: str | None = Field(default=None, max_length=1000)


class HazopRuleBatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rules: list[HazopRuleInput] = Field(min_length=1, max_length=5000)
    replace: bool = False


class HazopThresholdProposalRequest(BaseModel):
    """Ask RAG/LLM to propose, but never silently activate, a numeric limit."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    scenario_id: str = Field(
        default="*", max_length=40,
        validation_alias=AliasChoices("scenario_id", "scenarioId"),
    )
    tag_id: str = Field(
        min_length=1, max_length=80,
        validation_alias=AliasChoices("tag_id", "tagId"),
    )
    equipment_name: str = Field(
        default="", max_length=200,
        validation_alias=AliasChoices("equipment_name", "equipmentName"),
    )
    variable: str = Field(default="", max_length=200)
    unit: str = Field(default="", max_length=30)
    context: str = Field(default="", max_length=6000)
    model: str | None = Field(default=None, max_length=120)


class HazopThresholdCandidate(BaseModel):
    threshold_type: Literal["STATIC", "DELTA_START", "DELTA_PREV"]
    threshold_value: float
    compare_dir: Literal[">", ">=", "<", "<=", "이상", "초과", "이하", "미만"]
    unit: str = ""
    rationale: str = ""
    evidence_quote: str = ""
    doc_code: str = ""
    page: int | None = None
    threshold_basis: str = ""
    threshold_source: str = ""
    threshold_confidence: Literal["pending", "derived"] = "pending"


class HazopThresholdProposal(BaseModel):
    candidates: list[HazopThresholdCandidate] = Field(default_factory=list, max_length=20)
    decision_notes: str = ""


class DigitalTwinTagReading(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    tag_id: str = Field(
        min_length=1, max_length=80,
        validation_alias=AliasChoices("tag_id", "tagId"),
    )
    value: float | None = None
    unit: str = Field(default="", max_length=30)
    timestamp: datetime | None = None
    quality: str = Field(default="GOOD", max_length=40)
    baseline_start: float | None = Field(
        default=None,
        validation_alias=AliasChoices("baseline_start", "baselineStart"),
    )
    baseline_prev: float | None = Field(
        default=None,
        validation_alias=AliasChoices("baseline_prev", "baselinePrev"),
    )
    equipment_id: str | None = Field(
        default=None, max_length=80,
        validation_alias=AliasChoices("equipment_id", "equipmentId"),
    )
    equipment_name: str | None = Field(
        default=None, max_length=200,
        validation_alias=AliasChoices("equipment_name", "equipmentName"),
    )


class DigitalTwinEquipment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    equipment_id: str = Field(
        min_length=1, max_length=80,
        validation_alias=AliasChoices("equipment_id", "equipmentId", "id"),
    )
    name: str = Field(default="", max_length=200)
    equipment_type: str = Field(
        default="", max_length=100,
        validation_alias=AliasChoices("equipment_type", "equipmentType", "type"),
    )
    description: str = Field(default="", max_length=2000)
    tag_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tag_ids", "tagIds"),
    )


class DigitalTwinHazopRequest(BaseModel):
    """One digital-twin snapshot to evaluate against HAZOP and RAG evidence."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    station_id: str = Field(
        default="default", max_length=120,
        validation_alias=AliasChoices("station_id", "stationId"),
    )
    scenario_id: str = Field(
        default="*", max_length=40,
        validation_alias=AliasChoices("scenario_id", "scenarioId", "scenario"),
    )
    readings: list[DigitalTwinTagReading] = Field(default_factory=list, max_length=1000)
    equipment: list[DigitalTwinEquipment] = Field(default_factory=list, max_length=500)
    # A digital-twin instance may own the authoritative HAZOP catalogue (for
    # example the simulator's 205 rule rows).  Passing the snapshot here lets
    # the low-latency endpoint evaluate those rows without a database import or
    # a slow RAG lookup.  The normal endpoint keeps using the local DB when this
    # field is omitted.
    hazop_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=5000,
        validation_alias=AliasChoices("hazop_rules", "hazopRules", "rules"),
    )
    # Calculated by the digital twin from this same sensor snapshot before the
    # direct evaluation. SAGA relays these results; it does not invent them.
    impact_results: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=20,
        validation_alias=AliasChoices("impact_results", "impactResults"),
    )
    context_text: str = Field(
        default="", max_length=20000,
        validation_alias=AliasChoices("context_text", "contextText"),
    )
    condition: str = Field(default="", max_length=4000)
    interpret: bool = True
    generate_sop: bool = Field(
        default=True,
        validation_alias=AliasChoices("generate_sop", "generateSop"),
    )
    model: str | None = Field(default=None, max_length=120)
    max_staleness_seconds: int = Field(
        default=30,
        ge=1,
        le=86400,
        validation_alias=AliasChoices("max_staleness_seconds", "maxStalenessSeconds"),
    )


class HazopReference(BaseModel):
    label: str
    source_type: str
    status: Literal["indexed", "hazop_table", "catalog_only", "not_found"]
    doc_code: str = ""
    title: str = ""
    page: int | None = None
    excerpt: str = ""
    source_url: str | None = None


class HazopThresholdProposalResponse(BaseModel):
    scenario_id: str
    tag_id: str
    ready: bool = False
    requires_human_approval: bool = True
    candidates: list[HazopThresholdCandidate] = Field(default_factory=list)
    references: list[HazopReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class HazopHit(BaseModel):
    rule_id: int | None = None
    no: int | None = None
    scenario_id: str
    tag_id: str
    equipment_id: str | None = None
    equipment_name: str | None = None
    item_name: str = ""
    value: float | None = None
    unit: str = ""
    guide_word: str = ""
    threshold_value: float | None = None
    effective_threshold: float | None = None
    threshold_basis: str = ""
    threshold_source: str = ""
    threshold_confidence: Literal["verified", "derived", "unverified", "pending"] = "unverified"
    compare_dir: str = ""
    severity: str = "주의"
    severity_rank: int = 1
    risk_scenario: str = ""
    consequence: str = ""
    emergency_action: str = ""
    future_measure: str = ""
    standard_ref: str = ""
    source: str = "hazop_table"
    source_url: str | None = None


class HazopSop(BaseModel):
    answer: str
    source_status: Literal["hazop_grounded", "supplemented", "llm_general", "not_generated"]
    priority: Literal["normal", "attention", "warning", "emergency"] = "attention"
    immediate_actions: list[str] = Field(default_factory=list)
    isolation_evacuation: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    restart_requirements: list[str] = Field(default_factory=list)
    records_to_capture: list[str] = Field(default_factory=list)
    escalation: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[HazopReference] = Field(default_factory=list)


class DigitalTwinHazopResponse(BaseModel):
    evaluation_id: int | None = None
    station_id: str
    scenario_id: str
    evaluated_at: datetime
    status: Literal["NORMAL", "WARNING", "PARTIAL", "UNKNOWN"]
    worst_severity: str | None = None
    worst_rank: int = 0
    hit_count: int = 0
    hits: list[HazopHit] = Field(default_factory=list)
    unevaluated: list[str] = Field(default_factory=list)
    stale_tags: list[str] = Field(default_factory=list)
    missing_rule_tags: list[str] = Field(default_factory=list)
    evaluated_tag_count: int = 0
    requested_tag_count: int = 0
    state_key: str = ""
    sop: HazopSop | None = None
    data_quality: list[str] = Field(default_factory=list)
    monitoring_ready: bool = False
    threshold_gaps: list[str] = Field(default_factory=list)
    processing_ms: int | None = None
    impact_results: list[dict[str, Any]] = Field(default_factory=list)
