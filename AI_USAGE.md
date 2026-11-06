# AI usage

> Working notes through V2. The full write-up (achievements, evaluation results, limits) is completed in V3.

## AI stack

| Layer | Choice |
|---|---|
| LLM provider | Local models via Ollama (default: `qwen3:8b`) or Google Gemini, chosen with `LLM_PROVIDER` |
| Agent framework | LangChain v1 `create_agent` (tool calling) |
| Structured output | `ToolStrategy` over Pydantic schemas, with automatic retry on validation errors |
| Embeddings | `nomic-embed-text` (Ollama) or `gemini-embedding-001` |
| Vector database | Chroma (local, persistent) |
| PDF parsing | PyMuPDF + regex parsers with accounting-identity validation (LLM only as fallback) |
| Retrieval | Speaker-aware chunks; BM25 + vector search fused with reciprocal rank fusion |

## Reasoning approach

The master prompt makes the agent work numbers → trends → narrative → cross-check → forecast.
Numbers come only from the FinancialDataExtractorTool. Narrative comes only from transcript
excerpts retrieved by the QualitativeAnalysisTool.

## Guardrails

- Grounding rules in the prompt: only tool output, a source for every risk and opportunity, gaps recorded instead of guessed.
- Temperature 0 for extraction and analysis; 0.2 for the final synthesis.
- Financial figures parsed by rules and checked with 7 accounting identities per quarter
  (63/63 pass); the LLM never writes the per-quarter numbers in the final answer.
- Every quote is verified against the transcript chunks (≥ 80% word coverage); unsupported quotes
  are removed and reported; quarter and speaker are corrected from the source chunk.
- Forecast period checked against the latest quarter in the data.
- Pydantic validation on every LLM output; failed validation is sent back to the model.
- Gemini client retries (3) and 120 s timeouts; recursion limit of 40 agent steps.

## Development

This project was built with an AI coding assistant (Claude) for planning, code generation
and tests. Design decisions and verification are documented in the README and the release plan.

## Provider trade-off (found while building V1)

Gemini's free tier capped the project at 20 chat requests per model per day, too few for repeated
forecasts. Ollama was added as the default provider: no key, no quotas, data stays local. The cost is
slower generation and the weaker reasoning of an 8B model. Gemini remains one setting away.

## What V2 found (and why the design changed)

- **The LLM extractor was quietly wrong.** pdfplumber read "65,799" as "6 5,799" because the fact
  sheets draw a phantom space glyph over the first digit. The V1 LLM extraction then guessed Q4 FY26
  revenue as 70,694 instead of 70,698. Rules plus accounting checks fixed it, with no LLM calls.
- **One fact sheet draws its income statement as rotated text.** PyMuPDF reads it correctly.
- **Analyst detection needed two signals.** Moderators introduce analysts in different ways
  ("from the line of X", "question is from X"), so a positional fallback was added.
