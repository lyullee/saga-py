# SAGA-PY

Python/FastAPI safety-knowledge assistant for Korean gas standards, law and HAZOP analysis. SAGA-PY supports a standalone document-grounded chat and separate, direct APIs for a hydrogen-station digital twin's main conversation and selected-sensor analysis.

**Public source-only release:** private PDFs, local indexes/databases, credentials, generated evaluation responses and the former Java service are not included. Document-grounded use requires your own lawful corpus and provider key.

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md): architecture, retrieval, evidence validation, APIs, privacy, integration and limitations.
- [User and administrator manual](docs/USER_MANUAL.md): installation, indexing, Korean/English operation, diagnostics and maintenance.
- [Retrieval strategy](docs/RETRIEVAL_STRATEGY.md), [term aliases](docs/TERM_ALIAS_DICTIONARY.md), and [digital-twin HAZOP API](docs/DIGITAL_TWIN_HAZOP_API.md).

## Quick start

Requires Python 3.11+.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
# Put the real provider key in the local .env, never in .env.example.
.venv\Scripts\saga doctor
.venv\Scripts\saga index
.venv\Scripts\saga serve
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). `OPEN_AI_SERVICE_HUB_API_KEY` and `GROQ_API_KEY` are independent; select one provider manually. No automatic fallback occurs. The index command needs your own PDFs in the configured upload directory. A corpus-free installation can run the UI and general chat but cannot cite absent documents.

## Korean and English

The sidebar language switch changes the interface and the language of new answers. In the Korean standards/law mode, an English question is translated into a Korean retrieval query; the grounded answer is then rendered in English with citations, numbers and tags retained. English-language NREL/HIAD libraries keep the original English query. This adds model calls and latency. Verify exact requirements against the original Korean source. The digital twin's own language selector passes English directly to the isolated station/sensor assistant APIs and does not alter its consequence calculations.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Offline tests use mocks and fixtures. A network/corpus evaluation requires separate local documents and credentials. SAGA and the digital twin are independent repositories and normally listen on ports **8090** and **8000**, respectively.

SAGA output supports analysis and training. It does not control physical equipment or replace current official regulations, on-site verification or qualified emergency decision-making.
