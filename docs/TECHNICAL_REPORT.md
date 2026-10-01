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

Run `saga doctor` before indexing, `saga index` to build/update local data, `saga models` to verify provider reachability, and `saga serve` to start. Use `python -m pytest -q` for the unit/integration suite; network-backed answer quality evaluations require separate credentials/corpus and should not be mistaken for offline unit tests. Version the configuration and test representative Korean and English questions whenever retrieval or prompts change.

## 8. Limitations

RAG quality depends on the document set, OCR, currency of law/standards, metadata and model availability. The public clone has no included searchable corpus. English translation can alter nuance even when citation markers are preserved. HAZOP rule matches are scenario candidates and require actual instrument/field confirmation. The system is neither a legal authority nor an emergency control system. Use current official documents, site procedures and qualified reviewers for real decisions.
