"""QualitativeAnalysisTool.

RAG over the earnings-call transcripts:
  1. transcripts are indexed as speaker-aware chunks (speaker, role, section, page),
  2. every query is embedded once, then hybrid search (BM25 keywords + vectors, fused
     with reciprocal rank fusion) returns the best chunks from EACH quarter's call,
     optionally only management or only analyst turns,
  3. the LLM reads only those excerpts in ONE call and returns `QualitativeInsights`,
  4. every forward-looking statement is checked against the excerpts (grounding);
     unsupported quotes are removed, and quarter/speaker are corrected from the source chunk.
"""

import json
import logging
from typing import Literal

from langchain_core.tools import tool

from app.config import get_settings
from app.ingestion.index import get_hybrid_retriever
from app.ingestion.manifest import documents_of_type
from app.llm import get_chat_model
from app.schemas.financials import QualitativeInsights
from app.tools.grounding import verify_quote
from app.tools.quarters import normalize_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

MAX_QUERIES = 6

ANALYSIS_PROMPT = """You are an equity research analyst covering Tata Consultancy Services (TCS).
Below are excerpts from TCS earnings-call transcripts, grouped by quarter, retrieved for these topics:
{topics}

Each excerpt is labelled [quarter call, page] and starts with [speaker (role, section)]. Answers in
the Q&A may be preceded by the analyst question they respond to.

Using ONLY these excerpts:
- identify the recurring themes and the quarters each appears in,
- judge management's overall sentiment, noting how the tone changed between calls,
- list forward-looking statements (guidance, expectations, plans) made by MANAGEMENT, copied
  word for word from the excerpts (a sentence or two), each with its quarter and speaker,
- list risks and opportunities management mentioned.
Never put an analyst's words in management's mouth. If the excerpts do not cover something,
leave that list empty rather than guessing.

EXCERPTS:
{context}"""


def _transcript_quarters() -> list[str]:
    return sorted({d.quarter for d in documents_of_type("earnings_call_transcript")}, key=quarter_sort_key)


def retrieve(queries: list[str], quarter: str | None = None, role: str | None = None, k: int | None = None):
    """Best chunks for each query from each quarter's call (or one quarter), de-duplicated."""
    retriever = get_hybrid_retriever()
    k = k or get_settings().rag_chunks_per_quarter
    quarters = [normalize_quarter(quarter)] if quarter else _transcript_quarters()
    vectors = retriever.store.embeddings.embed_documents(queries)  # one embedding request for all queries
    seen, docs = set(), []
    for query, vector in zip(queries, vectors):
        for q in quarters:
            for d in retriever.search(query, vector, k=k, quarter=q, role=role):
                key = (d.metadata.get("doc_id"), d.metadata.get("turn"), d.metadata.get("chunk"))
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
    docs.sort(key=lambda d: (quarter_sort_key(d.metadata["quarter"]), d.metadata.get("turn", 0)))
    return docs


def ground_statements(insights: QualitativeInsights, docs) -> tuple[QualitativeInsights, int]:
    """Keep only forward-looking statements found in the excerpts; fix their quarter/speaker."""
    kept, dropped = [], 0
    for s in insights.forward_looking_statements:
        ok, src = verify_quote(s.statement, docs, normalize_quarter(s.quarter))
        if not ok:
            dropped += 1
            logger.info("Dropped unsupported statement: %s", s.statement[:100])
            continue
        s.quarter = src.metadata["quarter"]
        if src.metadata.get("role") == "management":
            s.speaker = src.metadata.get("speaker", s.speaker)
        kept.append(s)
    insights.forward_looking_statements = kept
    return insights, dropped


def analyze(queries: list[str], quarter: str | None = None, speaker_role: str | None = None) -> dict:
    queries = [q for q in queries if q.strip()][:MAX_QUERIES]
    if not queries:
        return {"error": "Provide at least one query."}
    docs = retrieve(queries, quarter, speaker_role)
    if not docs:
        return {"queries": queries, "error": "No transcript excerpts found. Run `python -m app.ingestion` first."}
    context = "\n\n".join(
        f"[{d.metadata['quarter']} call, page {d.metadata['page']}]\n{d.page_content}" for d in docs
    )
    llm = get_chat_model(temperature=0, for_tools=True).with_structured_output(QualitativeInsights)
    insights: QualitativeInsights = llm.invoke(
        ANALYSIS_PROMPT.format(topics="\n".join(f"- {q}" for q in queries), context=context)
    )
    insights, dropped = ground_statements(insights, docs)
    return {
        "queries": queries,
        "quarter_filter": quarter,
        "speaker_role_filter": speaker_role,
        "excerpts_used": len(docs),
        "grounding": {
            "statements_verified": len(insights.forward_looking_statements),
            "statements_removed_as_unsupported": dropped,
        },
        "insights": insights.model_dump(),
        "evidence": [
            {
                "quarter": d.metadata["quarter"],
                "page": d.metadata["page"],
                "speaker": d.metadata.get("speaker"),
                "role": d.metadata.get("role"),
                "excerpt": d.page_content[:300],
            }
            for d in docs
        ],
    }


@tool("qualitative_analysis")
def qualitative_analysis(
    queries: list[str],
    quarter: str | None = None,
    speaker_role: Literal["management", "analyst"] | None = None,
) -> str:
    """Semantic + keyword search and analysis across TCS earnings-call transcripts (RAG).

    Finds recurring themes, management sentiment (and how it changed between calls),
    verified forward-looking statements, and risks or opportunities management discussed.
    Pass ALL your topics in one call; excerpts are retrieved from every quarter's call.

    Args:
        queries: 1-6 focused topics, e.g. ["demand environment, deal wins and pipeline outlook",
            "operating margin drivers, headwinds and wage hikes",
            "AI, cloud and new growth opportunities",
            "risks, uncertainties and client spending caution"].
        quarter: Optional quarter label like "Q2 FY27" to search only that call.
        speaker_role: Optional "management" (guidance, outlook) or "analyst" (investor concerns).
    """
    try:
        return json.dumps(analyze(queries, quarter, speaker_role), indent=2, default=str)
    except Exception as exc:  # report to the agent instead of aborting the forecast
        logger.exception("qualitative_analysis failed for %r", queries)
        return json.dumps({"queries": queries, "error": f"{type(exc).__name__}: {exc}"})
