# Changelog

All notable changes to this project. Versions follow the release plan (V1 Foundation,
V2 Intelligence, V3 Production and submission).

## [2.0.0] - 2026-11-06 - V2 Intelligence

### Added
- **Rules-first financial extraction** (`factsheet_parser.py`): regex parsing of the consolidated
  Ind AS income statement and the performance-highlights page, with 7 accounting-identity checks
  per quarter; the LLM is used only as a fallback and its output is flagged unverified.
- New metrics: employee cost %, other income, PBT, USD and constant-currency growth, order-book
  TCV, headcount, attrition, cash conversion; employee-cost and headcount trends.
- `eval/ground_truth.json` and `python -m eval.extraction` (33/33 values match).
- **Speaker-aware transcript chunking**: turns tagged with speaker, role (management, analyst,
  moderator), section (prepared remarks, Q&A) and page; Q&A answers carry their question.
- **Hybrid retrieval**: BM25 + vector search fused with reciprocal rank fusion, per quarter,
  with optional speaker-role filter.
- **Grounding**: forward-looking statements and key quotes are verified against the transcripts
  (unsupported ones are removed, quarter and speaker corrected); per-quarter metrics are attached
  from the validated extractor instead of being written by the LLM; forecast period checked.
- MySQL tables `tool_calls` (every tool call with input, duration, output size, error) and
  `financial_metrics` (latest validated metrics per quarter).
- `GET /metrics` endpoint; tool-call trace in `GET /forecast/{request_id}`; `grounding` block in responses.

### Changed
- PDF text extraction moved from pdfplumber to PyMuPDF (fixes phantom-space digits and rotated tables).
- Agent writes `AgentForecast`; the API returns `ForecastResult` (agent output + validated metrics).
- Master prompt: parallel gathering, at most one follow-up search, stricter grounding rules.

### Fixed
- Q4 FY26 figures misread by the V1 LLM extractor (revenue 70,694 → 70,698; net profit 13,702 → 13,718; EPS 37.96 → 37.92).
- Analysts introduced as "question is from …" were tagged as management.

## [1.0.0] - 2026-11-06 - V1 Foundation

### Added
- Provider switch: local models via **Ollama** (default) or Google Gemini (`LLM_PROVIDER`).
- Batched multi-topic RAG with per-quarter retrieval (about 4 LLM calls per forecast).
- Free-tier protections for Gemini: request pacing, 429 retries, quota errors returned as HTTP 429.
- `python -m app.demo` one-command end-to-end run; database auto-created on startup.
- FastAPI service with `GET /health`, `POST /forecast` and `GET /forecast/{request_id}`.
- LangChain v1 tool-calling agent with a step-by-step master prompt and
  `ToolStrategy` structured output (`ForecastResult` Pydantic schema).
- **FinancialDataExtractorTool**: locates the consolidated income statement in TCS fact
  sheets, extracts per-quarter metrics with the LLM, recomputes margins, merges documents
  (newest wins), computes QoQ/YoY trends, and caches results.
- **QualitativeAnalysisTool**: Chroma vector search over earnings-call transcripts;
  returns themes, sentiment, forward-looking statements, risks and opportunities with evidence.
- Ingestion CLI (`python -m app.ingestion`) that downloads the documents in `data/manifest.json`
  and builds the vector index.
- MySQL logging of every request and response (success and error) in `request_logs`.
- Setup check CLI (`python -m app.check`).
- pytest suite (SQLite + scripted fake LLM; no network needed).
