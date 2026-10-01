# SAGA-PY — Technical Report

## 1. System purpose and boundaries

SAGA-PY is a Python/FastAPI retrieval and analysis service for Korean gas-safety standards, laws, procedures and HAZOP data. It offers a standalone web chat and dedicated digital-twin main/sensor assistant APIs. The public repository contains application code, configuration examples, tests and documentation. It **does not contain** the operator's private PDF corpus, local index/database, credentials, generated evaluation answers or the legacy Java application. A new checkout must supply its own licensed documents and API credentials before document-grounded answers are possible.

```text
PDFs / law snapshots ──indexer──► SQLite document + chunk + FTS5 tables
                                    └── hash n-gram vector index
                                                │
Question → mode routing → query normalization → hybrid retrieval/rerank
                                                │
                              grounded generation → citation validation
                                                │
                             optional answer review → response / SSE

Digital twin main/sensor → dedicated direct-assistant endpoints
Digital twin state/HAZOP → structured evaluation endpoints
```

The general `/api/chat` path must not be confused with the twin's direct assistant endpoints. A selected-sensor response needs current sensor/context data supplied by the twin; a general RAG query has no authoritative live sensor state. All LLM outputs are advisory and must be checked against the cited original and actual field conditions.

## 2. Code and dependencies

`python/saga/api.py` defines FastAPI routes, authentication gates, streaming and integration boundaries. `schemas.py` defines request/response contracts. `config.py` loads typed settings from environment or local `.env`; Windows user-level `GROQ_API_KEY` is re-read at start to avoid stale PyCharm process inheritance. `cli.py` exposes doctor/models/index/serve operations. `database.py` stores documents, chunks, metadata, logs and FTS structures. `indexer.py` extracts PDFs with PyMuPDF and constructs searchable chunks. `vector_index.py` uses a deterministic local hash n-gram representation. `rag.py` handles retrieval, evidence selection, answer generation and review. `text.py` normalizes terminology and response text. `hazop.py` handles rule import/evaluation. `law_api.py` and `external_sources.py` support optional public-source synchronization. `service_hub_client.py` and the Groq path use OpenAI-compatible APIs; provider choice is explicit, with no silent automatic fallback.

Requirements are Python 3.11+, FastAPI, OpenAI Python client, pydantic-settings, PyMuPDF, python-multipart and Uvicorn. SQLite FTS5 is supplied by the Python SQLite build. The service normally binds `127.0.0.1:8090`; a network deployment must set host, authentication, TLS/reverse proxy and access restrictions appropriate to its environment.

## 3. Indexing and evidence discipline

The indexer reads source PDFs from configured upload paths. The source library must be built on the deployment machine; the project does not ship those files. Document/chunk identity, page and section metadata enable source cards and links. Term aliases and `synonyms.txt` expand appropriate gas-safety wording, while ambiguous terms remain distinct. Hybrid retrieval combines FTS/BM25-style lexical matching with the local n-gram vector index. Query coverage and reranking narrow the context fed to the LLM. Where supported by the selected provider, an additional model stage checks relevance. The answer pipeline checks claim-to-source alignment, quotation and numeric/obligation wording; it may mark a claim as limited or switch to an ungrounded answer mode rather than fabricate a citation. These safeguards reduce, but cannot eliminate, hallucinations or obsolete standards.

For reproducibility, capture the corpus revision, index timestamp, alias dictionary, provider/model, prompt/options and returned source IDs. `docs/RETRIEVAL_STRATEGY.md` documents retrieval details. Local OCR recovery and law synchronization are separate optional workflows and may incur API calls. Source PDFs and generated answer/evaluation files stay outside the public repository.

## 4. Chat modes, provider and language

`ChatRequest` accepts `mode: "rag" | "chat"`, `answer_length`, provider and `language: "ko" | "en"`. In RAG mode, SAGA searches and validates the indexed documents. In general-chat mode it asks the selected model directly, without representing the answer as document-grounded. The provider selector chooses Service Hub or Groq manually; absence or failure of one key does **not** silently switch to the other. Streaming uses Server-Sent Events with status, draft/final and error phases. The browser renderer supports Markdown and citation links.

English mode keeps the source index authoritative. For Korean standards/law mode, the service translates an English question into a Korean retrieval query, runs the existing RAG/answer pipeline, then translates the verified answer into English while preserving citation markers, source identifiers, numbers and units. NREL/HIAD modes retain the original English query for their English-language sources. Translation requires additional LLM calls and can be slower. The English final answer for Korean standards is a translation of a Korean-grounded answer, not an independently validated English regulatory text. Source titles/quotes may remain Korean so that the original can be checked. For technical legal interpretation, compare the current official Korean source. In direct twin assistant requests, the language directive applies to the generated answer without changing the structured process data or consequence engine.

The UI selector persists in browser storage and re-labels dynamic controls. It intentionally leaves user text, citations and generated answer bodies untouched. Historical answers are not retranscribed after a language switch; send a new request to get output in the selected language.

## 5. Digital-twin integration

Distinct API families isolate responsibilities:

| Family | Purpose |
|---|---|
| `/api/integrations/digital-twin/main` (+ `/stream`) | User's station-level question and verified twin context. |
| `/api/integrations/digital-twin/sensor` (+ `/stream`) | Selected sensor question, value, related evidence and safeguards. |
| `/api/digital-twin/hazop/evaluate` | Structured rule evaluation; does not itself command a plant. |
| `/api/digital-twin/chat/direct` (+ `/stream`) | Legacy-compatible direct twin conversation path. |
| `/api/digital-twin/state/latest` | Most recent submitted twin state, if available. |

The twin sends `output_language` and current simulation evidence to the main/sensor integration request. Consequence estimates remain calculated in the twin and travel as data in the context; choosing English must never disable, replace or recompute HyRAM output. SAGA's direct path is designed for short, answer-focused replies, separate from the fuller document RAG workflow. The integration is still an advisory assistant: it cannot verify real valve feedback or issue real safety commands.

## 6. HTTP API and administration

Public/read operations include `/api/health`, `/api/config`, `/api/models`, `/api/chat`, `/api/chat/stream`, `/api/documents`, `/api/laws/catalog`, `/api/hazop/rules`, `/api/hazop/rules/monitoring-readiness`, and `/files/{document_id}`. Admin-gated operations include HAZOP import, law/source sync, PDF upload/reindex/delete. Set `SAGA_ADMIN_TOKEN` for a non-local deployment; inspect `require_admin` in `api.py` for the exact header contract. FastAPI exposes `/docs` when enabled by the deployment. Request/response fields are in `schemas.py` and OpenAPI. `ChatResponse.answer_mode` distinguishes sourced from LLM-only output, and `language` reports the requested output language.

## 7. Security, privacy and operations

The `.env.example` file is a template. Put real keys in an untracked `.env` or secure environment; never commit them. Index and upload directories contain source documents and may contain confidential text. Chat/feedback logs can contain user questions and excerpts; apply access control, retention and deletion appropriate to the documents. An externally reachable 8090 service should use TLS, limited network exposure, authentication and rate controls. Provider API requests may transfer question/evidence excerpts off-host under the selected provider's terms. The public code repository is intentionally source-only.

Run `saga doctor` before indexing, `saga index` to build/update local data, and `saga serve` to start. `saga models` lists **Service Hub** models and requires its key; it is not a Groq connectivity check. Use `python -m pytest -q` for the unit/integration suite; network-backed answer quality evaluations require separate credentials/corpus and should not be mistaken for offline unit tests. Version the configuration and test representative Korean and English questions whenever retrieval or prompts change.

## 8. Limitations

RAG quality depends on the document set, OCR, currency of law/standards, metadata and model availability. The public clone has no included searchable corpus. English translation can alter nuance even when citation markers are preserved. HAZOP rule matches are scenario candidates and require actual instrument/field confirmation. The system is neither a legal authority nor an emergency control system. Use current official documents, site procedures and qualified reviewers for real decisions.

## 9. Configuration reference

Values below are defaults from `python/saga/config.py`; deployment environment or local `.env` can override them. The public `.env.example` is a template and deliberately contains no valid key.

| Variable | Default / purpose |
|---|---|
| `SAGA_HOST`, `SAGA_PORT` | `127.0.0.1`, `8090` service binding. |
| `OPEN_AI_SERVICE_HUB_API_KEY` | No default; Service Hub credential. |
| `GROQ_API_KEY` | No default; Groq credential, selected manually. |
| `SAGA_SERVICE_HUB_BASE_URL` | Service Hub OpenAI-compatible `/v1` endpoint. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1`. |
| `SAGA_MODEL`, `SAGA_FAST_MODEL`, `SAGA_DIRECT_MODEL` | Main, fast and direct Service Hub model IDs. |
| `GROQ_MODEL`, `GROQ_FAST_MODEL`, `GROQ_DIRECT_MODEL` | Corresponding Groq model IDs. |
| `SAGA_UPLOAD_DIR` | Local `saga-uploads` PDF directory, excluded from Git. |
| `SAGA_DATABASE_PATH` | Local `data/saga.db`, excluded from Git. |
| `SAGA_RETRIEVAL_LIMIT` | 32 initial search hits (allowed 5–100). |
| `SAGA_CONTEXT_LIMIT` | 24 answer-context items (allowed 3–30). |
| `SAGA_MAX_CONTEXT_CHARS` | 42,000 context characters (allowed 5,000–100,000). |
| `SAGA_HYBRID_SEARCH_ENABLED` | `true` for lexical + hash n-gram vector search. |
| `SAGA_VECTOR_MODEL` | `hash-ngram-v1` local representation. |
| `SAGA_ANSWER_REVIEW_ENABLED` | `true`; review may add latency and provider calls. |
| `SAGA_ANSWER_LENGTH` | `standard`; per-request options include concise, detailed and very detailed. |
| `SAGA_LAW_API_OC` | Optional separate national law API credential. |
| `SAGA_ADMIN_TOKEN` | Empty by default; set and protect for externally reachable administration. |

The code's default host is localhost. A developer who copies a different older `.env` may still bind to all interfaces; inspect the local configuration and the actual bound address with OS network tools. `saga models` lists Service Hub models only. The browser provider selection is persisted client-side, while the server validates the corresponding credential on each request.

## 10. Document lifecycle and failure handling

`PdfIndexer` opens a PDF, extracts page text and metadata, builds section-oriented chunks and writes database rows. An unchanged file can be skipped during incremental indexing; `--force` rebuilds it. Poor text extraction is flagged for OCR rather than treated as reliable empty evidence. The optional `--service-hub-ocr` path uses a vision model and can incur external API cost. Law synchronization creates local PDF snapshots to make citations inspectable. NREL/HIAD acquisition and evaluation scripts are separate workflows; the public repository includes scripts but no fetched files or generated answer data.

At query time the pipeline normalizes the question and can reject a truly underspecified reference (“this” with no topic), while broad but meaningful safety questions proceed. It distinguishes standards, operations and incident knowledge modes to avoid mixing normative requirements with observed statistics or incident narratives. Search candidates are ranked and restricted before generation. Exact scope matters: a prohibition applying to one activity must not be broadened to an unrelated vehicle movement or facility duty. Answers with weak source match are labeled as limited or LLM-only. Secondary review is optional; a review output that introduces unsupported obligations or numbers is not automatically accepted. Citation validation is a post-generation guard and cannot compensate for an incomplete or outdated corpus.

A provider 401 should prompt credential/provider checks; a 429 may require waiting or reduced request rate. A retrieval miss should prompt source/index inspection, not a higher creativity setting. Streaming disconnections may leave the browser without a final event; clients should present the error and allow retry without claiming that the answer was validated. The English path adds a translation pass before standards retrieval and an English-rendering pass after the Korean result. The final translation is streamed as visible deltas where the selected provider supports streaming. The document citation cards preserve original metadata.

## 11. Interface and API contract examples

General document-grounded question:

```json
POST /api/chat
{"message":"What are the inspection conditions for hydrogen storage?",
 "provider":"groq", "language":"en", "mode":"rag",
 "knowledge_mode":"standards", "answer_length":"detailed"}
```

The response includes `answer`, `citations`, `answer_mode`, `mode`, `knowledge_mode`, `language`, `model`, conversation/log IDs and retrieval metadata. The stream variant sends named SSE events such as status, visible draft deltas, final answer and done/error. A caller must not infer that a citation exists when `answer_mode` is `llm_only`.

Digital-twin main integration:

```json
POST /api/integrations/digital-twin/main
{"question":"What should I check first?", "provider":"groq", "language":"en",
 "request_kind":"user_query", "context":{"station_status":"WARNING",
 "current_signals":{}, "impact_results":[]}}
```

The twin normally supplies a much richer context. The sensor endpoint uses a separate schema with `sensor_id`, selected-sensor `context`, `request_kind` and output language. These calls are short direct analyses and do not use the general chat's document search, conversation log or review pipeline. The twin's consequence engine provides `impact_results`; SAGA must not manufacture missing numbers. For full field definitions and response models, inspect `/openapi.json` or `schemas.py` on the version actually deployed.

## 12. Testing and change control

The public offline suite exercises settings, indexing, SQLite and hybrid retrieval, source handling, RAG quality guards, provider selection, HAZOP contracts, twin direct boundaries and English output. A single cross-code regression requires the operator's local private index and is skipped in a source-only checkout. A full acceptance run should add a separately governed corpus, fixed Korean/English query set, current documents, manual review of citations and failure-case checks for absent keys, OCR problems, false positives and incident-state contradictions. Preserve model IDs and corpus revision with each evaluation; a passing unit suite alone is not evidence of regulatory correctness.
