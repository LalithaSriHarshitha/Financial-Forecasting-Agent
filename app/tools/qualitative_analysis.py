"""QualitativeAnalysisTool.

RAG over the earnings-call transcripts:
  1. each query is embedded once,
  2. Chroma returns the best chunks for that query from EVERY quarter's call
     (per-quarter retrieval, so a comparison across calls is always possible),
  3. duplicates are removed and the excerpts are grouped by quarter,
  4. Gemini reads only those excerpts in ONE call and returns a structured
     `QualitativeInsights` object (themes, sentiment, forward-looking statements,
     risks, opportunities).

Batching several topics into one call keeps a forecast to a handful of LLM
requests, which matters on rate-limited (free-tier) API keys.
"""

import json
import logging

from langchain_core.tools import tool

from app.ingestion.index import get_shared_vector_store
from app.ingestion.manifest import documents_of_type
from app.llm import get_chat_model
from app.schemas.financials import QualitativeInsights
from app.tools.quarters import normalize_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

PER_QUARTER_K = 3  # chunks per query per quarter
MAX_QUERIES = 6

ANALYSIS_PROMPT = """You are an equity research analyst covering Tata Consultancy Services (TCS).
Below are excerpts from TCS earnings-call transcripts, grouped by quarter, retrieved for these topics:
{topics}

Using ONLY these excerpts:
- identify the recurring themes and the quarters each appears in,
- judge management's overall sentiment, noting how the tone changed between calls,
- list forward-looking statements (guidance, expectations, plans) as close to verbatim as possible,
  each with its quarter and the speaker if named,
- list risks and opportunities management mentioned.
If the excerpts do not cover something, leave that list empty rather than guessing.

EXCERPTS:
{context}"""


def _transcript_quarters() -> list[str]:
    return sorted({d.quarter for d in documents_of_type("earnings_call_transcript")}, key=quarter_sort_key)


def retrieve(queries: list[str], quarter: str | None = None, k: int = PER_QUARTER_K):
    """Best chunks for each query from each quarter (or one quarter), de-duplicated."""
    store = get_shared_vector_store()
    quarters = [normalize_quarter(quarter)] if quarter else _transcript_quarters()
    vectors = store.embeddings.embed_documents(queries)  # one embedding request for all queries
    seen, docs = set(), []
    for vector in vectors:
        for q in quarters:
            for d in store.similarity_search_by_vector(vector, k=k, filter={"quarter": q}):
                key = (d.metadata.get("doc_id"), d.metadata.get("page"), d.metadata.get("chunk"))
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
    docs.sort(key=lambda d: (quarter_sort_key(d.metadata["quarter"]), d.metadata["page"]))
    return docs


def analyze(queries: list[str], quarter: str | None = None) -> dict:
    queries = [q for q in queries if q.strip()][:MAX_QUERIES]
    if not queries:
        return {"error": "Provide at least one query."}
    docs = retrieve(queries, quarter)
    if not docs:
        return {
            "queries": queries,
            "error": "No transcript excerpts found. Run `python -m app.ingestion` first.",
        }
    context = "\n\n".join(
        f"[{d.metadata['quarter']} call, page {d.metadata['page']}]\n{d.page_content}" for d in docs
    )
    llm = get_chat_model(temperature=0, for_tools=True).with_structured_output(QualitativeInsights)
    insights: QualitativeInsights = llm.invoke(
        ANALYSIS_PROMPT.format(topics="\n".join(f"- {q}" for q in queries), context=context)
    )
    return {
        "queries": queries,
        "quarter_filter": quarter,
        "excerpts_used": len(docs),
        "insights": insights.model_dump(),
        "evidence": [
            {"quarter": d.metadata["quarter"], "page": d.metadata["page"], "excerpt": d.page_content[:300]}
            for d in docs
        ],
    }


@tool("qualitative_analysis")
def qualitative_analysis(queries: list[str], quarter: str | None = None) -> str:
    """Semantic search and analysis across TCS earnings-call transcripts (RAG).

    Finds recurring themes, management sentiment (and how it changed between calls),
    guidance and other forward-looking statements, and risks or opportunities management
    discussed. Pass ALL your topics in one call; excerpts are retrieved from every quarter's call.

    Args:
        queries: 1-6 focused topics, e.g. ["demand environment, deal wins and pipeline outlook",
            "operating margin drivers, headwinds and wage hikes",
            "AI, cloud and new growth opportunities",
            "risks, uncertainties and client spending caution"].
        quarter: Optional quarter label like "Q2 FY27" to search only that call.
    """
    try:
        return json.dumps(analyze(queries, quarter), indent=2, default=str)
    except Exception as exc:  # report to the agent instead of aborting the forecast
        logger.exception("qualitative_analysis failed for %r", queries)
        return json.dumps({"queries": queries, "error": f"{type(exc).__name__}: {exc}"})
