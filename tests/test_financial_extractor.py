import pytest

from app.schemas.financials import QuarterMetrics
from app.tools import financial_extractor as fe

INCOME_PAGE = """Ind AS Income Statement - Consolidated
₹ Crore Q2 FY26 Q1 FY27 Q2 FY27
Revenue 65,799 72,275 73,188
Profit Before Taxes & Other Income (Operating income) 16,565 17,317 17,553
Net Profit 12,904 13,849 13,884
Earnings per share in ₹ 35.67 38.28 38.37"""


def test_locates_income_statement_page():
    pages = ["Disclaimer", "Growth by Market: revenue by geography", INCOME_PAGE, "Balance sheet"]
    page_no, text = fe.locate_income_statement(pages)
    assert page_no == 3 and "Net Profit" in text


def test_no_income_statement_raises():
    with pytest.raises(ValueError):
        fe.locate_income_statement(["Disclaimer", "Thank you"])


def test_finalize_recomputes_margins():
    m = fe._finalize(QuarterMetrics(quarter="q2 fy27", revenue_cr=73188,
                                    operating_income_cr=17553, net_profit_cr=13884))
    assert m.quarter == "Q2 FY27"
    assert m.operating_margin_pct == pytest.approx(23.98, abs=0.01)
    assert m.net_margin_pct == pytest.approx(18.97, abs=0.01)


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


def test_newest_document_wins(monkeypatch):
    class Doc:
        def __init__(self, i):
            self.id, self.exists, self.filename = i, True, f"{i}.pdf"

    docs = [Doc("old"), Doc("new")]
    outputs = {
        "old": {"source": "old", "page": 1, "quarters": [{"quarter": "Q1 FY27", "revenue_cr": 1.0,
                "operating_margin_pct": None, "net_profit_cr": None}]},
        "new": {"source": "new", "page": 2, "quarters": [{"quarter": "Q1 FY27", "revenue_cr": 2.0,
                "operating_margin_pct": None, "net_profit_cr": None}]},
    }
    monkeypatch.setattr(fe, "documents_of_type", lambda _: docs)
    monkeypatch.setattr(fe, "extract_from_document", lambda d: outputs[d.id])
    out = fe.get_financial_metrics()
    assert out["metrics"][0]["revenue_cr"] == 2.0
    assert out["sources"]["Q1 FY27"] == "new, page 2"
