"""FinancialDataExtractorTool.

Reads TCS quarterly fact sheets (PDF), finds the consolidated income-statement
table, and extracts key metrics per quarter with Gemini structured output.

Pipeline per document:
  1. pdfplumber -> page text
  2. score pages to locate the consolidated income statement (rules, no LLM)
  3. LLM reads only that page and fills the `ExtractedStatement` schema
  4. derived values (margins) are recomputed in Python, and results are cached
Results from several documents are merged (the newest document wins, since it
carries the latest restated figures) and quarter-over-quarter trends are computed.
"""

import json
import logging

from langchain_core.tools import tool

from app.config import get_settings
from app.ingestion.manifest import DocumentEntry, documents_of_type
from app.ingestion.pdf import read_pages
from app.llm import get_chat_model
from app.schemas.financials import ExtractedStatement, QuarterMetrics
from app.tools.quarters import normalize_quarter, previous_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a meticulous financial analyst reading a page from {title}.
The page contains TCS's consolidated income statement in INR crore. Columns are fiscal
quarters (for example 'Q2 FY26', 'Q1 FY27', 'Q2 FY27'); some tables also show the same
quarters again as '% of revenue' columns - ignore those percentage columns except to read
the operating margin.

For EVERY quarter column (absolute INR values), extract:
- revenue_cr: the 'Revenue' line.
- operating_income_cr: 'Profit Before Taxes & Other Income (Operating income)' or 'Operating income'.
- operating_margin_pct: operating income as % of revenue (read the % column if present).
- net_profit_cr: the final 'Net Profit' / 'Net income attributable to shareholders' line.
- eps_inr: 'Earnings per share in ₹'.

Rules: numbers only (no commas or units); use null if a value is not on the page;
never estimate or invent values; write quarters as 'Q<n> FY<yy>'. Put caveats such as
'excludes exceptional items' in notes.

PAGE TEXT:
---
{page}
---"""


def _score_page(text: str) -> int:
    t = text.lower()
    score = 0
    if "income statement" in t:
        score += 5
    if "ind as" in t and "consolidated" in t and "income statement" in t:
        score += 10
    for kw in ("revenue", "net profit", "earnings per share", "operating income", "employee cost"):
        if kw in t:
            score += 2
    return score


def locate_income_statement(pages: list[str]) -> tuple[int, str]:
    """Return (1-based page number, text) of the page most likely to hold the income statement."""
    if not pages:
        raise ValueError("document has no pages")
    best = max(range(len(pages)), key=lambda i: _score_page(pages[i]))
    if _score_page(pages[best]) < 7:
        raise ValueError("no income-statement page found")
    return best + 1, pages[best]


def _finalize(m: QuarterMetrics) -> QuarterMetrics:
    m.quarter = normalize_quarter(m.quarter)
    if m.revenue_cr:
        if m.operating_income_cr is not None:
            m.operating_margin_pct = round(m.operating_income_cr / m.revenue_cr * 100, 2)
        if m.net_profit_cr is not None:
            m.net_margin_pct = round(m.net_profit_cr / m.revenue_cr * 100, 2)
    return m


def _cache_path(doc: DocumentEntry):
    return get_settings().processed_dir / "metrics" / f"{doc.id}.json"


def extract_from_document(doc: DocumentEntry, use_cache: bool = True) -> dict:
    """Extract all quarter columns from one fact sheet. Returns a JSON-serialisable dict."""
    cache = _cache_path(doc)
    mtime = doc.path.stat().st_mtime
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("source_mtime") == mtime:
            return cached

    page_no, page_text = locate_income_statement(read_pages(doc.path))
    llm = get_chat_model(temperature=0).with_structured_output(ExtractedStatement)
    statement: ExtractedStatement = llm.invoke(
        EXTRACTION_PROMPT.format(title=doc.title, page=page_text)
    )
    result = {
        "doc_id": doc.id,
        "source": doc.title,
        "page": page_no,
        "source_mtime": mtime,
        "notes": statement.notes,
        "quarters": [_finalize(q).model_dump() for q in statement.quarters],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Extracted %d quarter(s) from %s p.%d", len(statement.quarters), doc.id, page_no)
    return result


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or not old:
        return None
    return round((new - old) / old * 100, 2)


def compute_trends(metrics: dict[str, dict]) -> list[dict]:
    """QoQ and YoY changes for each quarter that has comparison data."""
    trends = []
    for q in sorted(metrics, key=quarter_sort_key):
        cur = metrics[q]
        row = {"quarter": q}
        for label, offset in (("qoq", 1), ("yoy", 4)):
            prev = metrics.get(previous_quarter(q, offset) or "")
            if not prev:
                continue
            row[f"revenue_growth_{label}_pct"] = _pct_change(cur["revenue_cr"], prev["revenue_cr"])
            row[f"net_profit_growth_{label}_pct"] = _pct_change(
                cur["net_profit_cr"], prev["net_profit_cr"]
            )
            if cur["operating_margin_pct"] is not None and prev["operating_margin_pct"] is not None:
                row[f"operating_margin_change_{label}_bps"] = round(
                    (cur["operating_margin_pct"] - prev["operating_margin_pct"]) * 100
                )
        if len(row) > 1:
            trends.append(row)
    return trends


def get_financial_metrics(quarters: list[str] | None = None) -> dict:
    """Merge metrics from all available fact sheets (newest document wins)."""
    merged: dict[str, dict] = {}
    sources: dict[str, str] = {}
    errors: list[str] = []
    for doc in documents_of_type("financial_report"):  # sorted oldest -> newest
        if not doc.exists:
            errors.append(f"{doc.filename} not found in data/raw/")
            continue
        try:
            extracted = extract_from_document(doc)
        except Exception as exc:
            logger.exception("Extraction failed for %s", doc.id)
            errors.append(f"{doc.id}: {exc}")
            continue
        for q in extracted["quarters"]:
            merged[q["quarter"]] = q
            sources[q["quarter"]] = f"{extracted['source']}, page {extracted['page']}"

    wanted = {normalize_quarter(q) for q in quarters} if quarters else None
    ordered = sorted(merged, key=quarter_sort_key)
    return {
        "unit": "INR crore (EPS in INR, margins in %)",
        "metrics": [merged[q] for q in ordered if not wanted or q in wanted],
        "trends": [t for t in compute_trends(merged) if not wanted or t["quarter"] in wanted],
        "sources": {q: sources[q] for q in ordered if not wanted or q in wanted},
        "errors": errors,
    }


@tool("financial_data_extractor")
def financial_data_extractor(quarters: list[str] | None = None) -> str:
    """Extract key financial metrics for TCS from its quarterly financial reports (fact sheets).

    Returns per-quarter revenue, operating income, operating margin, net profit, net margin
    and EPS (INR crore), plus quarter-over-quarter and year-over-year growth, with sources.

    Args:
        quarters: Optional list of quarter labels like ["Q1 FY27", "Q2 FY27"].
            Leave empty to get every quarter available in the reports.
    """
    try:
        return json.dumps(get_financial_metrics(quarters), indent=2)
    except Exception as exc:  # report to the agent instead of aborting the forecast
        logger.exception("financial_data_extractor failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
