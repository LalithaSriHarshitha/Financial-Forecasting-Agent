# Changelog

All notable changes to this project. Versions follow the release plan (V1 Foundation,
V2 Intelligence, V3 Production and submission).

## [1.0.0] - 2026-11-06 - V1 Foundation

### Added
- FastAPI service with `GET /health`, `POST /forecast` and `GET /forecast/{request_id}`.
- LangChain v1 tool-calling agent on Google Gemini with a step-by-step master prompt and
  `ToolStrategy` structured output (`ForecastResult` Pydantic schema).
- **FinancialDataExtractorTool**: locates the consolidated income statement in TCS fact
  sheets, extracts per-quarter metrics with Gemini, recomputes margins, merges documents
  (newest wins), computes QoQ/YoY trends, and caches results.
- **QualitativeAnalysisTool**: Chroma + Gemini embeddings over earnings-call transcripts;
  returns themes, sentiment, forward-looking statements, risks and opportunities with evidence.
- Ingestion CLI (`python -m app.ingestion`) that downloads the documents in `data/manifest.json`
  and builds the vector index.
- MySQL logging of every request and response (success and error) in `request_logs`.
- Setup check CLI (`python -m app.check`).
- pytest suite (SQLite + scripted fake LLM; no network needed).
