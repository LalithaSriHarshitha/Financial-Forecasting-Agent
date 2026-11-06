"""FinancialDataExtractorTool.

Pipeline per fact sheet (rules first, LLM only as a fallback):
  1. PyMuPDF -> page text
  2. locate the consolidated Ind AS income statement and the performance-highlights page
  3. parse every line item with regex (`factsheet_parser`) and validate the accounting
     identities (cost lines sum to total, revenue - costs = operating income,
     PBT - tax = PAT, ...). If parsing or a check fails, the LLM reads that page and
     its output is flagged `verified: false`.
  4. results are cached per file version and stored in the MySQL `financial_metrics` table

Across documents: each fact sheet repeats earlier quarters, so the newest document
wins and any disagreement between documents (a restatement) is reported.
"""

import json
import logging

from langchain_core.tools import tool

from app.config import get_settings
from app.ingestion.manifest import DocumentEntry, documents_of_type
from app.ingestion.pdf import read_pages
from app.llm import get_chat_model
from app.schemas.financials import ExtractedStatement, QuarterMetrics
from app.tools.factsheet_parser import (
    find_highlights_page,
    find_income_statement_page,
    parse_highlights,
    parse_income_statement,
)
from app.tools.quarters import normalize_quarter, previous_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

PARSER_VERSION = 2  # bump to invalidate cached extractions

LLM_FALLBACK_PROMPT = """You are a meticulous financial analyst reading a page from {title}.
The page contains TCS's consolidated income statement in INR crore. Columns are fiscal
quarters (for example 'Q2 FY26', 'Q1 FY27', 'Q2 FY27'), followed by the same quarters as
'% of revenue' - ignore the percentage columns.

For EVERY quarter column extract: revenue_cr ('Revenue'), operating_income_cr ('Profit Before
Taxes & Other Income'), net_profit_cr (final 'Net Profit'), eps_inr ('Earnings per share').
Numbers only, null if absent, never estimate. Quarters as 'Q<n> FY<yy>'.

PAGE TEXT:
---
{page}
---"""


def _ratio(num: float | None, den: float | None) -> float | None:
    return round(num / den * 100, 2) if num is not None and den else None


def _rows_from_rules(doc: DocumentEntry, page_no: int, stmt) -> list[dict]:
    rows = []
    for q in stmt.quarters:
        v = stmt.values[q]
        m = QuarterMetrics(
            quarter=q,
            revenue_cr=v.get("revenue_cr"),
            employee_cost_cr=v.get("employee_cost_cr"),
            employee_cost_pct=_ratio(v.get("employee_cost_cr"), v.get("revenue_cr")),
            operating_income_cr=v.get("operating_income_cr"),
            operating_margin_pct=_ratio(v.get("operating_income_cr"), v.get("revenue_cr")),
            other_income_cr=v.get("other_income_cr"),
            profit_before_tax_cr=v.get("profit_before_tax_cr"),
            net_profit_cr=v.get("net_profit_cr"),
            net_margin_pct=_ratio(v.get("net_profit_cr"), v.get("revenue_cr")),
            eps_inr=v.get("eps_inr"),
            source=f"{doc.title}, page {page_no}",
            extraction_method="rules",
            verified=all(c["passed"] for c in stmt.checks if c["quarter"] == q),
        )
        rows.append(m.model_dump())
    return rows


def _rows_from_llm(doc: DocumentEntry, page_no: int, page_text: str) -> list[dict]:
    llm = get_chat_model(temperature=0, for_tools=True).with_structured_output(ExtractedStatement)
    statement: ExtractedStatement = llm.invoke(LLM_FALLBACK_PROMPT.format(title=doc.title, page=page_text))
    rows = []
    for r in statement.quarters:
        m = QuarterMetrics(
            quarter=normalize_quarter(r.quarter),
            revenue_cr=r.revenue_cr,
            operating_income_cr=r.operating_income_cr,
            operating_margin_pct=_ratio(r.operating_income_cr, r.revenue_cr),
            net_profit_cr=r.net_profit_cr,
            net_margin_pct=_ratio(r.net_profit_cr, r.revenue_cr),
            eps_inr=r.eps_inr,
            source=f"{doc.title}, page {page_no}",
            extraction_method="llm_fallback",
            verified=False,
        )
        rows.append(m.model_dump())
    return rows


def _cache_path(doc: DocumentEntry):
    return get_settings().processed_dir / "metrics" / f"{doc.id}.json"


def extract_from_document(doc: DocumentEntry, use_cache: bool = True) -> dict:
    """Extract every quarter column (and highlights) from one fact sheet."""
    cache = _cache_path(doc)
    mtime = doc.path.stat().st_mtime
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("source_mtime") == mtime and cached.get("parser_version") == PARSER_VERSION:
            return cached

    pages = read_pages(doc.path)
    idx = find_income_statement_page(pages)
    if idx is None:
        raise ValueError(f"no quarterly income statement found in {doc.filename}")
    stmt = parse_income_statement(pages[idx])
    if stmt.valid:
        rows, method = _rows_from_rules(doc, idx + 1, stmt), "rules"
    else:
        failed = [c for c in stmt.checks if not c["passed"]]
        logger.warning("Rules parsing failed validation for %s (%s); using LLM fallback", doc.id, failed[:2])
        rows, method = _rows_from_llm(doc, idx + 1, pages[idx]), "llm_fallback"

    h_idx = find_highlights_page(pages)
    highlights = parse_highlights(pages[h_idx]) if h_idx is not None else None

    result = {
        "doc_id": doc.id,
        "source": doc.title,
        "page": idx + 1,
        "parser_version": PARSER_VERSION,
        "source_mtime": mtime,
        "method": method,
        "checks": stmt.checks,
        "quarters": rows,
        "highlights": highlights,
        "highlights_page": h_idx + 1 if h_idx is not None else None,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Extracted %d quarter(s) from %s p.%d via %s", len(rows), doc.id, idx + 1, method)
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
            row[f"revenue_growth_{label}_pct"] = _pct_change(cur.get("revenue_cr"), prev.get("revenue_cr"))
            row[f"net_profit_growth_{label}_pct"] = _pct_change(cur.get("net_profit_cr"), prev.get("net_profit_cr"))
            for fld in ("operating_margin_pct", "employee_cost_pct"):
                if cur.get(fld) is not None and prev.get(fld) is not None:
                    row[f"{fld.removesuffix('_pct')}_change_{label}_bps"] = round((cur[fld] - prev[fld]) * 100)
            if label == "qoq" and cur.get("headcount") and prev.get("headcount"):
                row["headcount_change_qoq"] = int(cur["headcount"] - prev["headcount"])
        if len(row) > 1:
            trends.append(row)
    return trends


def _merge(extractions: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """Newest document wins; report quarters whose figures differ between documents."""
    merged: dict[str, dict] = {}
    restatements = []
    for ex in extractions:  # oldest -> newest
        for row in ex["quarters"]:
            q = row["quarter"]
            old = merged.get(q)
            if old and old.get("revenue_cr") and row.get("revenue_cr"):
                for fld in ("revenue_cr", "operating_income_cr", "net_profit_cr"):
                    if old.get(fld) is not None and row.get(fld) is not None and abs(old[fld] - row[fld]) > 1:
                        restatements.append({"quarter": q, "field": fld, "earlier": old[fld],
                                             "latest": row[fld], "latest_source": row["source"]})
            merged[q] = {**(old or {}), **{k: v for k, v in row.items() if v is not None}}
        h = ex.get("highlights")
        if h and h.get("quarter"):
            target = merged.setdefault(h["quarter"], {"quarter": h["quarter"]})
            for k, v in h.items():
                if k != "quarter" and v is not None:
                    target[k] = int(v) if k == "headcount" else v
    return merged, restatements


_tables_ready = False


def _save_to_db(rows: list[dict]) -> None:
    global _tables_ready
    try:
        from app.db.repository import upsert_financial_metrics
        from app.db.session import init_db

        if not _tables_ready:  # also works from CLI scripts that never start the API
            init_db()
            _tables_ready = True
        upsert_financial_metrics(rows)
    except Exception as exc:  # the tool still works without the database
        logger.warning("Could not store financial metrics in the database: %s", str(exc).splitlines()[0][:200])


def get_financial_metrics(quarters: list[str] | None = None) -> dict:
    """Merged metrics from all available fact sheets, with trends and validation summary."""
    extractions, errors = [], []
    for doc in documents_of_type("financial_report"):  # sorted oldest -> newest
        if not doc.exists:
            errors.append(f"{doc.filename} not found in data/raw/")
            continue
        try:
            extractions.append(extract_from_document(doc))
        except Exception as exc:
            logger.exception("Extraction failed for %s", doc.id)
            errors.append(f"{doc.id}: {exc}")

    merged, restatements = _merge(extractions)
    _save_to_db(list(merged.values()))
    wanted = {normalize_quarter(q) for q in quarters} if quarters else None
    ordered = [q for q in sorted(merged, key=quarter_sort_key) if not wanted or q in wanted]
    checks = [c for ex in extractions for c in ex.get("checks", [])]
    return {
        "unit": "INR crore (EPS in INR; margins, growth and attrition in %; TCV in USD bn)",
        "metrics": [merged[q] for q in ordered],
        "trends": [t for t in compute_trends(merged) if t["quarter"] in ordered],
        "validation": {
            "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": len(checks),
            "methods": {ex["doc_id"]: ex["method"] for ex in extractions},
            "restatements": restatements,
        },
        "errors": errors,
    }


@tool("financial_data_extractor")
def financial_data_extractor(quarters: list[str] | None = None) -> str:
    """Extract TCS financial metrics from the quarterly fact sheets (validated, with sources).

    Returns per quarter: revenue, employee cost (% of revenue), operating income and margin,
    net profit and margin, EPS (INR crore), plus USD and constant-currency growth, order-book
    TCV, headcount, attrition and cash conversion; QoQ/YoY trends; and a validation summary.

    Args:
        quarters: Optional list like ["Q1 FY27", "Q2 FY27"]. Leave empty for every quarter.
    """
    try:
        return json.dumps(get_financial_metrics(quarters), indent=2)
    except Exception as exc:  # report to the agent instead of aborting the forecast
        logger.exception("financial_data_extractor failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
