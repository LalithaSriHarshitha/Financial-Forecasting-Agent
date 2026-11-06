"""Deterministic (rules-based) parser for TCS quarterly fact sheets.

Two pages carry what the forecast needs:
  * "Ind AS Income Statement - Consolidated": one row per line item, three quarter
    columns in INR crore followed by the same quarters as % of revenue.
  * "<Quarter> Performance Highlights": USD and constant-currency growth, order-book
    TCV, headcount and attrition.

Every figure is read by regex from the page text, then the income statement is
validated with accounting identities (e.g. revenue - total expenditure = operating
income). The LLM is used only as a fallback when parsing or validation fails.
"""

import re
from dataclasses import dataclass, field

QUARTER_RE = re.compile(r"\bQ([1-4])\s*FY\s*'?(\d{2})\b")
NUMBER_RE = re.compile(r"\(?-?[\d,]*\d(?:\.\d+)?\)?")

# Line items: field name -> regex matched at the start of the (stripped) row label.
LINE_ITEMS: dict[str, str] = {
    "revenue_cr": r"Revenue\b",
    "employee_cost_cr": r"a\)\s*Employee Costs?",
    "total_expenditure_cr": r"Total Expenditure",
    "operating_income_cr": r"Profit Before Taxes\s*&\s*Other Income",
    "other_income_cr": r"Other income",
    "profit_before_tax_cr": r"Profit Before Taxes(?!\s*&)",
    "tax_cr": r"Provision For Taxes",
    "profit_after_tax_cr": r"Profit After Taxes",
    "non_controlling_cr": r"Non-controlling interests",
    "net_profit_cr": r"Net Profit",
    "eps_inr": r"Earnings per share",
}
COST_ITEMS = [r"a\)\s*Employee", r"b\)\s*Cost of equipment", r"c\)\s*Other Operating", r"d\)\s*Depreciation"]

TOLERANCE_CR = 3.0  # rounding in the published table


def _to_number(token: str) -> float:
    neg = token.startswith("(") and token.endswith(")") or token.startswith("-")
    value = float(token.strip("()-").replace(",", ""))
    return -value if neg else value


def quarter_label(q: str, fy: str) -> str:
    return f"Q{q} FY{int(fy) % 100:02d}"


@dataclass
class ParsedStatement:
    quarters: list[str]
    values: dict[str, dict[str, float]]  # quarter -> field -> value
    pct_of_revenue: dict[str, dict[str, float]]
    checks: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.quarters) and all(c["passed"] for c in self.checks)


def find_income_statement_page(pages: list[str]) -> int | None:
    """0-based index of the quarterly consolidated Ind AS income statement page."""
    for i, text in enumerate(pages):
        head = text[:400].lower()
        if "income statement" in head and "ind as" in head and "consolidated" in head:
            if len(QUARTER_RE.findall(text[:800])) >= 2:  # quarterly, not the annual (FY) page
                return i
    return None


def _row_numbers(line: str, label_re: str) -> list[float] | None:
    m = re.match(rf"\s*{label_re}", line, re.IGNORECASE)
    if not m:
        return None
    rest = line[m.end():]
    rest = re.sub(r"^[^\d(\-]*", "", rest)  # drop the remainder of the label text
    return [_to_number(t) for t in NUMBER_RE.findall(rest)]


def parse_income_statement(text: str) -> ParsedStatement:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = next((ln for ln in lines if len(QUARTER_RE.findall(ln)) >= 2), "")
    found = [quarter_label(q, fy) for q, fy in QUARTER_RE.findall(header)]
    quarters = list(dict.fromkeys(found))  # header lists the quarters twice (INR and %)
    n = len(quarters)
    values: dict[str, dict[str, float]] = {q: {} for q in quarters}
    pct: dict[str, dict[str, float]] = {q: {} for q in quarters}
    costs: dict[str, float] = {q: 0.0 for q in quarters}
    cost_rows = 0

    for line in lines if quarters else []:
        for fld, label_re in LINE_ITEMS.items():
            nums = _row_numbers(line, label_re)
            already_set = fld in values[quarters[0]]  # keep the first matching row
            if nums is None or len(nums) < n or already_set:
                continue
            for i, q in enumerate(quarters):
                values[q][fld] = nums[i]
                if len(nums) >= 2 * n:
                    pct[q][fld] = nums[n + i]
            break
        for cost_re in COST_ITEMS:
            nums = _row_numbers(line, cost_re)
            if nums and len(nums) >= n:
                cost_rows += 1
                for i, q in enumerate(quarters):
                    costs[q] += nums[i]

    stmt = ParsedStatement(quarters=quarters, values=values, pct_of_revenue=pct)
    stmt.checks = _validate(stmt, costs if cost_rows == 4 else None)
    return stmt


def _check(name: str, quarter: str, expected: float | None, actual: float | None, tol: float) -> dict:
    passed = expected is not None and actual is not None and abs(expected - actual) <= tol
    return {"check": name, "quarter": quarter, "expected": expected, "actual": actual, "passed": passed}


def _validate(stmt: ParsedStatement, costs: dict[str, float] | None) -> list[dict]:
    """Accounting identities that must hold for every quarter column."""
    checks = []
    for q in stmt.quarters:
        v = stmt.values[q]
        g = v.get
        required = ("revenue_cr", "operating_income_cr", "net_profit_cr", "eps_inr")
        checks.append({"check": "required fields present", "quarter": q,
                       "missing": [f for f in required if f not in v],
                       "passed": all(f in v for f in required)})
        if costs is not None and g("total_expenditure_cr") is not None:
            checks.append(_check("sum of cost lines = total expenditure", q,
                                 g("total_expenditure_cr"), costs[q], TOLERANCE_CR))
        if g("revenue_cr") is not None and g("total_expenditure_cr") is not None:
            checks.append(_check("revenue - total expenditure = operating income", q,
                                 g("operating_income_cr"), g("revenue_cr") - g("total_expenditure_cr"), TOLERANCE_CR))
        if g("operating_income_cr") is not None and g("other_income_cr") is not None:
            checks.append(_check("operating income + other income = PBT", q,
                                 g("profit_before_tax_cr"), g("operating_income_cr") + g("other_income_cr"), TOLERANCE_CR))
        if g("profit_before_tax_cr") is not None and g("tax_cr") is not None:
            checks.append(_check("PBT - tax = PAT", q,
                                 g("profit_after_tax_cr"), g("profit_before_tax_cr") - g("tax_cr"), TOLERANCE_CR))
        if g("profit_after_tax_cr") is not None and g("non_controlling_cr") is not None:
            checks.append(_check("PAT - non-controlling interests = net profit", q,
                                 g("net_profit_cr"), g("profit_after_tax_cr") - g("non_controlling_cr"), TOLERANCE_CR))
        published = stmt.pct_of_revenue[q].get("operating_income_cr")
        if published is not None and g("revenue_cr"):
            checks.append(_check("operating margin matches published %", q,
                                 published, round(g("operating_income_cr") / g("revenue_cr") * 100, 1), 0.15))
    return checks


# ---------------- Performance highlights page ----------------

def find_highlights_page(pages: list[str]) -> int | None:
    for i, text in enumerate(pages):
        if re.search(r"Q[1-4]\s*FY\s*\d{2}\s+Performance Highlights", text[:300]):
            return i
    return None


def _growth(text: str, prefix: str) -> dict[str, float]:
    """Parse 'up 1.2% QoQ | down 0.6% YoY' / 'Flat QoQ' after a given prefix."""
    line = next((ln for ln in text.splitlines() if re.search(prefix, ln, re.IGNORECASE)), "")
    out = {}
    for period in ("QoQ", "YoY"):
        m = re.search(rf"(up|down)\s+([\d.]+)%\s*{period}", line, re.IGNORECASE)
        if m:
            out[period] = float(m.group(2)) * (-1 if m.group(1).lower() == "down" else 1)
        elif re.search(rf"flat\s*{period}", line, re.IGNORECASE):
            out[period] = 0.0
    return out


def _first(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1).replace(",", "")) if m else None


def parse_highlights(text: str) -> dict:
    """Operational signals for the quarter named in the page title."""
    title = QUARTER_RE.search(text[:300])
    usd = _growth(text, r"USD Revenue")
    cc = _growth(text, r"Constant currency")
    return {
        "quarter": quarter_label(*title.groups()) if title else None,
        "usd_revenue_mn": _first(r"USD Revenue of \$\s*([\d,]+(?:\.\d+)?)\s*Mn", text),
        "usd_growth_qoq_pct": usd.get("QoQ"),
        "usd_growth_yoy_pct": usd.get("YoY"),
        "cc_growth_qoq_pct": cc.get("QoQ"),
        "cc_growth_yoy_pct": cc.get("YoY"),
        "order_book_tcv_usd_bn": _first(r"Order book TCV at \$\s*([\d.]+)\s*Bn", text),
        "headcount": _first(r"Closing headcount:\s*([\d,]+)", text),
        "attrition_ltm_pct": _first(r"attrition at\s*([\d.]+)%", text),
        "cfo_to_net_profit_pct": _first(r"Cash flow from operations at\s*([\d.]+)%", text),
    }
