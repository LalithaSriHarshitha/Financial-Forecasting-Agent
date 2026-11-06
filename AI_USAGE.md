# AI usage

> Working notes for V1. The full write-up (achievements, evaluation results, limits) is completed in V3.

## AI stack

| Layer | Choice |
|---|---|
| LLM provider | Google Gemini (chat model configurable via `GEMINI_MODEL`) |
| Agent framework | LangChain v1 `create_agent` (tool calling) |
| Structured output | `ToolStrategy` over Pydantic schemas, with automatic retry on validation errors |
| Embeddings | `gemini-embedding-001` |
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
