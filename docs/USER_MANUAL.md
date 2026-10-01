# SAGA-PY — User and Administrator Manual

## 1. Install from the public repository

Use Python 3.11 or newer. In a new checkout, create a virtual environment, activate it and install the package:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit the **local `.env`**, never `.env.example`, with a valid `OPEN_AI_SERVICE_HUB_API_KEY` and/or `GROQ_API_KEY`. The provider is selected manually in the UI or request. Optional settings include `SAGA_MODEL`, `SAGA_FAST_MODEL`, `GROQ_MODEL`, `SAGA_ANSWER_REVIEW_ENABLED`, `SAGA_UPLOAD_DIR`, `SAGA_DATABASE_PATH`, `SAGA_HOST`, `SAGA_PORT` and `SAGA_ADMIN_TOKEN`. See [Technical Report](TECHNICAL_REPORT.md) and `python/saga/config.py` for defaults. Do not publish `.env` or document files.

Run:

```powershell
.venv\Scripts\saga doctor
.venv\Scripts\saga models
.venv\Scripts\saga serve
```

Open `http://127.0.0.1:8090`. If the provider key changed while PyCharm was open, restart the SAGA process. A 401 `invalid_api_key` is a provider credential/configuration problem, not a retrieval failure. Check the provider actually selected and the effective environment. No automatic provider failover is performed.

## 2. Build the local document index

This public repository contains no private PDF corpus or prebuilt database. Supply documents you may lawfully use to the configured upload directory, then run `.venv\Scripts\saga index`. `saga index --force` rebuilds all configured documents. Optional OCR-assisted recovery may require a vision model and extra API calls. `saga doctor` reports index and configuration readiness. In the browser, **Documents** shows available files and source links. If RAG search returns no relevant source, confirm PDFs were indexed and inspect their extracted text/metadata before changing prompts. Keep the corpus and database backed up separately from Git.

An administrator may upload, reindex or delete via the API/UI if `SAGA_ADMIN_TOKEN` is configured. Law/source synchronization is optional and uses separate credentials and APIs; it is not required for the base chat service. A generated index can contain substantial excerpts from source material and should receive the same access controls as the PDFs.

## 3. Ask a question

Choose **RAG evidence** when the answer must refer to indexed standards, laws or procedures. Choose **General chat** when no document citation is required. Select Service Hub or Groq and the answer length before sending. The answer panel shows progress, final Markdown, sources and any warning that the answer lacks direct documentary support. Open a cited source and compare its actual text when a limit, legal obligation, exception or numerical threshold matters. An “LLM-only” notice means the answer is general model knowledge, not verified corpus evidence.

You can select one of the suggested prompts or type your own. Ask a narrow, explicit question for the quickest focused answer: equipment, condition, desired action and document scope help retrieval. Avoid including secrets or personal/site-sensitive data in a cloud-provider request. Feedback on an answer is recorded locally and helps identify retrieval failures; it does not automatically amend the index or model.

## 4. Use Korean or English

Use the **한국어 / English** selector in the sidebar. The setting is remembered by the browser. New English questions in standards/law RAG mode are translated into Korean for retrieval; SAGA validates the answer against the Korean source, then renders it in English while preserving citation markers, values and units. NREL/HIAD modes retain the English query for English sources. English is supported for normal use, but translation requires additional calls and can respond more slowly. Citations lead to the original source. When regulatory nuance matters, verify the Korean wording. Existing messages are not rewritten when you toggle the selector; resend the question for a new-language answer. The twin's monitor/remote has its own language selector and passes that setting to the dedicated SAGA assistant APIs.

## 5. Digital-twin questions

The digital twin's **main SAGA panel** takes station-level questions; its **selected-sensor panel** receives a specific tag, value, relevant scenarios and consequence context. These channels are distinct from this SAGA website's general RAG chat. If the twin reports an active alarm, check the source tag, quality, operation state, injected fault and calculated consequence status. An LLM safety response is advice; actions are executed only through the digital twin's separate **virtual safety** controls, and real plant action always follows site authority. For an English twin response, switch the twin to English before asking.

## 6. HAZOP and administrator workflows

The HAZOP rules API supports inspection, monitoring-readiness checks, threshold proposals, and administrator-gated import. A rule's warning/alarm state is a modelled condition, not proof of a physical leak. Confirm sensor quality, other detectors, process path and the actual incident input. Rule evaluation and the direct assistant remain separate; changing a prompt does not change the HAZOP database. After importing rules, rerun readiness checks and the unit tests, then test representative normal, warning and critical conditions.

## 7. Diagnostics

| Observation | Likely checks |
|---|---|
| 401 invalid API key | Selected provider, key spelling, environment/.env precedence, server restart. |
| RAG says no documents | Configured upload path, `saga index`, PDF extraction/OCR, database path and file permissions. |
| English answer slow | Translation adds model calls; try concise answer or a narrower question. |
| English answer has Korean source names | Original titles/citations are intentionally preserved for traceability. |
| Stream pauses | Provider latency, retrieval/review stages, server log and client network connection. |
| Twin response lacks current state | Confirm twin-to-SAGA integration route and that current job context was supplied; general SAGA chat has no live plant connection. |
| Citation seems wrong | Open the PDF/page and verify current revision; report feedback and reindex if the source changed. |

## 8. Maintenance and safety

Run `.venv\Scripts\python -m pytest -q` after code/config changes. Back up the corpus, SQLite database, aliases and HAZOP rules before reindex or import. Rotate provider/admin keys if disclosed. When exposing the service beyond localhost, use an authenticated HTTPS reverse proxy, narrow network access and an explicit data-retention policy. Do not treat generated text, English translation or a virtual incident response as a substitute for current official standards or competent field judgment.

For architecture, API families, privacy and limits, see [Technical Report](TECHNICAL_REPORT.md). For the twin protocol, see [Digital Twin HAZOP API](DIGITAL_TWIN_HAZOP_API.md).
