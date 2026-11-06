"""FinancialDataExtractorTool tests on real fact-sheet page text (tests/fixtures)."""

from pathlib import Path

import pytest

from app.tools import factsheet_parser as fp
from app.tools import financial_extractor as fe

FIX = Path(__file__).parent / "fixtures"


def page(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "fixture, quarters, latest",
    [
        ("Q4FY26_income_statement.txt", ["Q4 FY25", "Q3 FY26", "Q4 FY26"], (70698, 17870, 13718, 37.92)),
        ("Q1FY27_income_statement.txt", ["Q1 FY26", "Q4 FY26", "Q1 FY27"], (72275, 17317, 13849, 38.28)),
        ("Q2FY27_income_statement.txt", ["Q2 FY26", "Q1 FY27", "Q2 FY27"], (73188, 17553, 13884, 38.37)),
    ],
)
def test_rules_parser_reads_real_income_statements(fixture, quarters, latest):
    st = fp.parse_income_statement(page(fixture))
    assert st.quarters == quarters
    v = st.values[quarters[-1]]
    assert (v["revenue_cr"], v["operating_income_cr"], v["net_profit_cr"], v["eps_inr"]) == latest
    assert st.valid, [c for c in st.checks if not c["passed"]]
    assert len(st.checks) == 21  # 7 accounting checks x 3 quarters


def test_validation_catches_a_misread_number():
    text = page("Q2FY27_income_statement.txt").replace("73,188", "73,108", 1)  # simulate an OCR/LLM misread
    st = fp.parse_income_statement(text)
    failed = {c["check"] for c in st.checks if not c["passed"]}
    assert not st.valid
    assert "revenue - total expenditure = operating income" in failed


def test_highlights_parser():
    h = fp.parse_highlights(page("Q4FY26_highlights.txt"))
    assert h["quarter"] == "Q4 FY26"
    assert h["cc_growth_qoq_pct"] == 1.2 and h["cc_growth_yoy_pct"] == -0.6  # "down 0.6% YoY"
    assert h["order_book_tcv_usd_bn"] == 12.0 and h["headcount"] == 584519 and h["attrition_ltm_pct"] == 13.7
    assert fp.parse_highlights(page("Q1FY27_highlights.txt"))["usd_growth_qoq_pct"] == 0.0  # "Flat QoQ"


def test_page_finders_skip_annual_statement():
    annual = "Ind AS Income Statement - Consolidated\nFY25 FY26 FY25 FY26\nRevenue 255,324 267,021"
    pages = ["cover", annual, page("Q4FY26_income_statement.txt"), page("Q4FY26_highlights.txt")]
    assert fp.find_income_statement_page(pages) == 2
    assert fp.find_highlights_page(pages) == 3


class _Doc:
    def __init__(self, tmp_path, text, id_="doc"):
        self.id, self.title, self.filename = id_, f"{id_} fact sheet", f"{id_}.pdf"
        self.path = tmp_path / self.filename
        self.path.write_text("x")
        self.exists = True
        self._text = text


def _patch_pages(monkeypatch, text):
    monkeypatch.setattr(fe, "read_pages", lambda path: ["cover", text, page("Q2FY27_highlights.txt")])


def test_extract_uses_rules_and_merges_highlights(monkeypatch, tmp_path):
    monkeypatch.setattr(fe.get_settings(), "data_dir", tmp_path)
    _patch_pages(monkeypatch, page("Q2FY27_income_statement.txt"))
    ex = fe.extract_from_document(_Doc(tmp_path, ""), use_cache=False)
    assert ex["method"] == "rules"
    merged, restatements = fe._merge([ex])
    q2 = merged["Q2 FY27"]
    assert q2["verified"] and q2["order_book_tcv_usd_bn"] == 9.6 and q2["headcount"] == 598056
    assert q2["operating_margin_pct"] == pytest.approx(23.98, abs=0.01)
    assert restatements == []


def test_llm_fallback_when_validation_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(fe.get_settings(), "data_dir", tmp_path)
    _patch_pages(monkeypatch, page("Q2FY27_income_statement.txt").replace("73,188", "73,108", 1))
    called = {}

    def fake_llm(doc, page_no, text):
        called["yes"] = True
        return [fe.QuarterMetrics(quarter="Q2 FY27", revenue_cr=73188, extraction_method="llm_fallback").model_dump()]

    monkeypatch.setattr(fe, "_rows_from_llm", fake_llm)
    ex = fe.extract_from_document(_Doc(tmp_path, ""), use_cache=False)
    assert called and ex["method"] == "llm_fallback"
    assert ex["quarters"][0]["verified"] is False


def test_restatement_between_documents_is_reported():
    old = {"quarters": [{"quarter": "Q1 FY27", "revenue_cr": 72275.0, "source": "old"}], "highlights": None}
    new = {"quarters": [{"quarter": "Q1 FY27", "revenue_cr": 72300.0, "source": "new"}], "highlights": None}
    merged, restatements = fe._merge([old, new])
    assert merged["Q1 FY27"]["revenue_cr"] == 72300.0
    assert restatements[0]["field"] == "revenue_cr"


def test_compute_trends_qoq_and_yoy():
    def q(label, rev, op, net):
        return {"quarter": label, "revenue_cr": rev, "operating_margin_pct": op, "net_profit_cr": net}

    metrics = {
        "Q2 FY26": q("Q2 FY26", 65799, 25.17, 12904),
        "Q1 FY27": q("Q1 FY27", 72275, 23.96, 13849),
        "Q2 FY27": q("Q2 FY27", 73188, 23.98, 13884),
    }
    trends = {t["quarter"]: t for t in fe.compute_trends(metrics)}
    assert trends["Q2 FY27"]["revenue_growth_qoq_pct"] == pytest.approx(1.26, abs=0.01)
    assert trends["Q2 FY27"]["revenue_growth_yoy_pct"] == pytest.approx(11.23, abs=0.01)
    assert trends["Q2 FY27"]["operating_margin_change_qoq_bps"] == 2
