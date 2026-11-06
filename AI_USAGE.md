# AI usage

This document answers the five questions in the assignment: the AI stack and reasoning approach,
the tools and models used, what the AI achieved end to end, the guardrails and evaluation, and
the limits and trade-offs. Every number below was measured on the live system on 6 Nov 2026
unless marked otherwise; the evaluation report is in [`eval/report.md`](eval/report.md).

---

## 1. AI stack and reasoning approach

| Layer | Choice | Why |
|---|---|---|
| LLM | Google **Gemini** (`gemini-3.6-flash`, default) or local models via **Ollama** (`qwen3:8b`), set by `LLM_PROVIDER` | Gemini: fast, strong tool calling, free tier. Ollama: no key, private, no quotas |
| Agent framework | **LangChain v1** `create_agent` (LangGraph tool-calling loop) | Native function calling, parallel tool calls, structured output |
| Function calling | 3 tools: `financial_data_extractor`, `qualitative_analysis`, `market_data` | One tool per kind of evidence |
| Structured output | `ToolStrategy(AgentForecast)` over Pydantic; invalid output is sent back with the validation error | Predictable, machine-readable JSON |
| Embeddings | `gemini-embedding-001` (or `nomic-embed-text` on Ollama) | Same key as the LLM; avoids a 2 GB PyTorch install on Windows |
| Vector DB | **Chroma** (local, persistent), one index per embedding model | No server to install; metadata filters (quarter, role) |
| Keyword search | **BM25** (`rank-bm25`), fused with vectors by reciprocal rank fusion | Exact names and terms (MHP, BSNL, TCV) |
| PDF parsing | **PyMuPDF** + regex parsers; LLM only as fallback | See section 5 for why |
| OCR | Not needed: all six documents have a text layer. A scanned page would fail validation and go to the LLM fallback | |
| Market data | `yfinance` (Yahoo Finance, `TCS.NS`) | Free, no key |
| Storage | MySQL 8.0 via SQLAlchemy | Assignment requirement; audit trail |

**Reasoning approach.** The agent follows a fixed analyst workflow, set by the master prompt
(`app/agent/prompts.py`):

1. **Gather**: all three tools in parallel, in one turn.
2. **Trends**: CC growth, margin drivers, TCV, headcount, attrition.
3. **Follow-up**: at most one targeted search, for example management-only or a single quarter.
4. **Cross-check**: management's claims against the numbers.
5. **Forecast**: structured directions plus reasoning and confidence.

Division of labour:
- **Deterministic code** does what must be exact: reading the numbers, arithmetic and trends, valuation, and checking quotes.
- **The LLM** does what needs judgement: themes, tone, weighing evidence, and writing the outlook.

---

## 2. Tools and models: how each works

| Tool | AI involvement | Deterministic parts |
|---|---|---|
| FinancialDataExtractorTool | None on the happy path. If a page fails validation, an LLM fallback (temperature 0) fills a small schema, flagged `verified: false` | PDF text; page location; regex for 11 line items + % columns + 9 operational fields; 7 accounting identities per quarter; merge across documents; QoQ/YoY trends |
| QualitativeAnalysisTool | 1 LLM call per invocation: reads retrieved excerpts → `QualitativeInsights` | Speaker-turn parsing; role and section tagging; chunking; hybrid retrieval per quarter; checking each statement against the excerpts |
| MarketDataTool | None | yfinance quote; P/E from validated trailing EPS; 5-minute cache |
| Agent | 3–4 LLM calls: plan and gather → (optional follow-up) → final structured answer | Attaches metrics and market snapshot, checks quotes, checks forecast period, logs everything |

A forecast uses about **4–5 chat requests** and 1–2 embedding requests. It takes about 100 s on Gemini's free tier.

---

## 3. What the AI achieved end to end

**Data sources retrieved.** Six documents covering the last three quarters, found through
[Screener.in](https://www.screener.in/company/TCS/consolidated/#documents) and listed in `data/manifest.json`:
- 3 TCS fact sheets (Q4 FY26, Q1 FY27, Q2 FY27) from tcs.com. tcs.com blocks scripted downloads (HTTP 403), so these are downloaded manually; the log prints the URL.
- 3 earnings-call transcripts from BSE, downloaded automatically.

**Metrics extracted.**
- 7 quarters of consolidated figures (Q4 FY25–Q2 FY27), because each fact sheet carries three quarters.
- 11 income-statement fields per quarter, plus USD/CC growth, order-book TCV, headcount, attrition and cash conversion for the three latest quarters.
- **33/33 values match** the hand-typed ground truth (`python -m eval.extraction`).
- **63/63 accounting checks pass**, with no LLM calls.

**Transcripts.**
- 277 speaker-aware chunks from 229 speaker turns.
- Exactly the five TCS executives are tagged as management in all three calls; analysts are tagged from the moderator's introductions.

**Synthesis quality** (live V2 run, `sample_output.json`):
- **Forecast:** Q3 FY27.
- **Judgement:** cautiously optimistic sentiment; revenue steady; slight margin compression.
- **Drivers identified:**
  - Q3 seasonal furloughs;
  - roughly 50 bps of dilution from the MHP acquisition;
  - AI revenue now above 10% of the total;
  - TCV of $9.6 bn.
- **Grounding:**
  - 6/6 quotes verified against the transcripts;
  - every risk and opportunity cites its quarter and speaker.
- **Reasoning:** the agent made one extra search on its own, limited to management in Q2 FY27, on furloughs and MHP.
- **Faithfulness:** 31/31 checks passed (`eval/report.md`).

---

## 4. Guardrails and evaluation

**Prompting strategy**
- A role, the fiscal calendar, explicit numbered steps, and a fixed set of 4 transcript queries, so runs are comparable.
- A cap of one follow-up search, to keep cost and latency predictable.
- Strict grounding rules: use only tool output; copy quotes word for word; give every risk and opportunity a source; write gaps into `data_gaps`.
- No investment advice and no price targets.
- The LLM does not write the numbers; code attaches them.

**Output control and retries**
- **Structured output:** Pydantic schemas with `Literal` enums for directions, sentiment and confidence. Validation errors are returned to the model so it can retry.
- **Rate-limit handling:**
  - a shared rate limiter for chat calls (`LLM_REQUESTS_PER_MINUTE`);
  - pacing for embeddings (`EMBED_REQUESTS_PER_MINUTE`);
  - up to 6 client retries with backoff on 429 and 503 responses.
- **Error handling:**
  - tools return errors as JSON instead of crashing the agent, so the forecast continues and records the gap;
  - quota errors return HTTP 429, and requests that run past `FORECAST_TIMEOUT_S` return HTTP 504.
- **Ollama limits:** a cap on output tokens and a per-call timeout stop runaway generations.

**Grounding checks (post-generation, deterministic)**
1. **Accounting identities on every extracted quarter.** If any check fails, the LLM fallback is used and the figures are flagged unverified.
2. **Statement and quote verification.** At least 80% of a quote's content words must appear in one transcript chunk. Unsupported quotes are removed and counted; the quarter and speaker are corrected from the source chunk. This was tested: an invented quote is removed, and a quote mislabelled as Q1 FY27 is corrected to Q2 FY27, Samir Seksaria.
3. **Forecast period.** It must be the quarter after the latest analysed one; otherwise it is corrected.
4. **Per-quarter numbers and the market snapshot are attached from tool data**, never generated.

**Observability.** Every request and response is stored in `request_logs`, every tool call (input, duration, output size, error) in `tool_calls`, and the latest validated figures in `financial_metrics`. Each response includes a `usage` block with LLM calls and tokens, and a `grounding` block.

**Evaluation** (`python -m eval.run` → `eval/report.md`)

| Evaluation | Method | Result |
|---|---|---|
| Extraction accuracy | 33 ground-truth values typed from the fact sheets | 33/33 = 100% |
| Retrieval quality | Hit@3 on 12 paraphrased questions (no key terms), hybrid vs vector-only vs BM25-only | see `eval/report.md` |
| Output faithfulness | 31 deterministic checks on the saved forecast (metrics, quotes, sources, period, no advice) | 31/31 |
| Consistency | Agreement of `revenue_direction`, `margin_direction` and `confidence` over N full runs (`--consistency 3`) | see `eval/report.md` (costs about 15 requests) |
| Unit and integration tests | 35 pytest tests: real fact-sheet text, the agent graph with a scripted LLM, API, timeouts, quotas, concurrency | all pass |

---

## 5. Limits, trade-offs and mitigations

| Problem found | Impact | Mitigation |
|---|---|---|
| **The LLM misread numbers.** pdfplumber read "65,799" as "6 5,799" (the fact sheets draw a phantom space over each number's first digit), and the V1 LLM extractor then guessed Q4 FY26 revenue as 70,694 instead of 70,698 | Wrong numbers, delivered confidently | Rules-first parsing with PyMuPDF plus 7 accounting identities; the LLM is only a flagged fallback; ground-truth evaluation |
| One fact sheet draws its income statement as **rotated text** | Unreadable for pdfplumber | PyMuPDF reads it correctly |
| **Gemini free tier: 20 requests per model per day**, plus frequent 503 "high demand" errors | V1 used ~10 calls per forecast and ran out of quota | Batched topics (4–5 calls), shared rate limiter, retries, clear HTTP 429; `GEMINI_TOOL_MODEL` can spread load across a second model's quota |
| **Ollama on a laptop CPU** ran for over 30 minutes without finishing | Not practical without a GPU | Gemini is the default; Ollama is optional, with output caps, timeouts and progress logging |
| **Parallel tool calls** created Chroma clients at the same time (thread-safety bug) | Crashed the request | One shared client behind a lock; a regression test runs 6 threads |
| **Moderators phrase analyst introductions differently** | An analyst was mislabelled as management | Two signals: the introduction phrase, plus who speaks right after the moderator in Q&A |
| tcs.com blocks scripted downloads (HTTP 403) | Can't fetch fact sheets automatically | Manifest with exact URLs and file names; the log explains the manual step |
| Quote verification uses word coverage, not meaning | A paraphrase that keeps 80% of the words passes | Strict enough to catch invented quotes and wrong quarters; the prompt asks for word-for-word quotes |
| A qualitative forecast can't be scored against the future yet | Accuracy is unknown until Q3 FY27 results (Jan 2027) | Directions are structured, so the forecast can be scored once Q3 FY27 is reported |
| Free-tier quota limits evaluation runs | Consistency is measured on few runs | `--consistency N` is configurable; run more on a paid key |

---

## 6. How AI was used to build this project

The project was built with an AI coding assistant (Claude):
- planning the three-version release, with acceptance gates;
- writing the code, tests and documentation;
- debugging from the real error logs: quotas, Chroma threading, Windows console behaviour.

Every design change was driven by evidence from real runs on the actual TCS documents:
- the phantom-space digits;
- the rotated table;
- the quota limits;
- the analyst mislabel.

The extraction ground truth was typed from the fact-sheet pages independently of the parser. All
results above come from live runs, not from the assistant's claims.
