# AI usage

> Working notes for V1. The full write-up (achievements, evaluation results, limits) is completed in V3.

## AI stack

| Layer | Choice |
|---|---|
| LLM provider | Local models via Ollama (default: `qwen3:8b`) or Google Gemini, chosen with `LLM_PROVIDER` |
| Agent framework | LangChain v1 `create_agent` (tool calling) |
| Structured output | `ToolStrategy` over Pydantic schemas, with automatic retry on validation errors |
| Embeddings | `nomic-embed-text` (Ollama) or `gemini-embedding-001` |
| Vector database | Chroma (local, persistent) |
| PDF parsing | pdfplumber (OCR fallback planned for V2) |

## Reasoning approach

The master prompt makes the agent work numbers → trends → narrative → cross-check → forecast.
Numbers come only from the FinancialDataExtractorTool. Narrative comes only from transcript
excerpts retrieved by the QualitativeAnalysisTool.

## Guardrails in V1

- Grounding rules in the prompt: only tool output, a source for every risk and opportunity, gaps recorded instead of guessed.
- Temperature 0 for extraction and analysis; 0.2 for the final synthesis.
- Margins recomputed in Python from extracted values.
- Pydantic validation on every LLM output; failed validation is sent back to the model.
- Gemini client retries (3) and 120 s timeouts; recursion limit of 40 agent steps.

## Development

This project was built with an AI coding assistant (Claude) for planning, code generation
and tests. Design decisions and verification are documented in the README and the release plan.

## Provider trade-off (found while building V1)

Gemini's free tier capped the project at 20 chat requests per model per day, too few for repeated
forecasts. Ollama was added as the default provider: no key, no quotas, data stays local. The cost is
slower generation and the weaker reasoning of an 8B model. Gemini remains one setting away.
