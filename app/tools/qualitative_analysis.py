"""QualitativeAnalysisTool.

RAG over the earnings-call transcripts: semantic search in Chroma, then Gemini
reads only the retrieved excerpts and returns themes, sentiment and
forward-looking statements as a structured `QualitativeInsights` object.
"""

import json
import logging

from langchain_core.tools import tool

from app.ingestion.index import get_shared_vector_store
from app.llm import get_chat_model
from app.schemas.financials import QualitativeInsights
from app.tools.quarters import normalize_quarter

logger = logging.getLogger(__name__)

TOP_K = 8

ANALYSIS_PROMPT = """You are an equity research analyst covering Tata Consultancy Services (TCS).
Below are excerpts from TCS earnings-call transcripts retrieved for the question:
"{query}"

Using ONLY these excerpts:
- identify the recurring themes (and the quarters each appears in),
- judge management's overall sentiment and explain why,
- list forward-looking statements (guidance, expectations, plans) as close to verbatim as possible,
- list risks and opportunities management mentioned.
If the excerpts do not cover something, leave that list empty rather than guessing.

EXCERPTS:
{context}"""


def retrieve(query: str, quarter: str | None = None, k: int = TOP_K):
    flt = {"quarter": normalize_quarter(quarter)} if quarter else None
    return get_shared_vector_store().similarity_search(query, k=k, filter=flt)


def analyze(query: str, quarter: str | None = None) -> dict:
    docs = retrieve(query, quarter)
    if not docs:
        return {
            "query": query,
            "error": "No transcript excerpts found. Run `python -m app.ingestion` first.",
        }
    context = "\n\n".join(
        f"[{d.metadata['quarter']} call, page {d.metadata['page']}]\n{d.page_content}" for d in docs
    )
    llm = get_chat_model(temperature=0).with_structured_output(QualitativeInsights)
    insights: QualitativeInsights = llm.invoke(ANALYSIS_PROMPT.format(query=query, context=context))
    return {
        "query": query,
        "quarter_filter": quarter,
        "insights": insights.model_dump(),
        "evidence": [
            {
                "quarter": d.metadata["quarter"],
                "page": d.metadata["page"],
                "excerpt": d.page_content[:400],
            }
            for d in docs
        ],
    }


@tool("qualitative_analysis")
def qualitative_analysis(query: str, quarter: str | None = None) -> str:
    """Semantic search and analysis across TCS earnings-call transcripts (RAG).

    Use it to find recurring themes, management sentiment, guidance and other
    forward-looking statements, and risks or opportunities management discussed.
    Call it several times with focused queries (e.g. "demand outlook and deal pipeline",
    "margin headwinds and wage hikes", "AI revenue and opportunities", "risks and uncertainties").

    Args:
        query: What to look for in the transcripts.
        quarter: Optional quarter label like "Q2 FY27" to restrict the search to one call.
    """
    try:
        return json.dumps(analyze(query, quarter), indent=2, default=str)
    except Exception as exc:  # report to the agent instead of aborting the forecast
        logger.exception("qualitative_analysis failed for %r", query)
        return json.dumps({"query": query, "error": f"{type(exc).__name__}: {exc}"})
