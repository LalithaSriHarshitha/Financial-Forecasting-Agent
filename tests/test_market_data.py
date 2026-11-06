import importlib
import json

md = importlib.import_module("app.tools.market_data")

QUOTE = {"currency": "INR", "lastPrice": 3150.0, "previousClose": 3100.0, "yearHigh": 3500.0,
         "yearLow": 2900.0, "fiftyDayAverage": 3050.0, "twoHundredDayAverage": 3200.0,
         "yearChange": -0.05, "marketCap": 11.4e12}
METRICS = {"metrics": [{"quarter": q, "eps_inr": e} for q, e in
                       [("Q2 FY26", 35.67), ("Q3 FY26", 37.14), ("Q4 FY26", 37.92), ("Q1 FY27", 38.28), ("Q2 FY27", 38.37)]]}


def _setup(monkeypatch, quote=QUOTE):
    fe = importlib.import_module("app.tools.financial_extractor")
    monkeypatch.setattr(fe, "get_financial_metrics", lambda quarters=None: METRICS)
    monkeypatch.setattr(md, "_cache", {"at": 0.0, "data": None})
    monkeypatch.setattr(md.get_settings(), "market_data_enabled", True)

    def fetch(symbol):
        if isinstance(quote, Exception):
            raise quote
        return quote

    monkeypatch.setattr(md, "fetch_quote", fetch)


def test_snapshot_uses_validated_ttm_eps(monkeypatch):
    _setup(monkeypatch)
    snap = md.get_market_snapshot(force=True)
    assert snap["available"] and snap["eps_ttm_quarters"] == ["Q3 FY26", "Q4 FY26", "Q1 FY27", "Q2 FY27"]
    assert snap["eps_ttm_inr"] == 151.71
    assert snap["pe_ttm"] == round(3150 / 151.71, 2)
    assert snap["day_change_pct"] == 1.61 and snap["pct_below_52w_high"] == 10.0
    assert snap["one_year_change_pct"] == -5.0 and snap["market_cap_inr_cr"] == 1140000


def test_failure_is_reported_not_raised(monkeypatch):
    _setup(monkeypatch, quote=ConnectionError("Yahoo unreachable"))
    out = json.loads(md.market_data.invoke({}))
    assert out["available"] is False and "Yahoo unreachable" in out["error"]


def test_snapshot_is_cached(monkeypatch):
    _setup(monkeypatch)
    calls = []
    real = md.fetch_quote
    monkeypatch.setattr(md, "fetch_quote", lambda s: calls.append(s) or real(s))
    md.get_market_snapshot()
    md.get_market_snapshot()
    assert len(calls) == 1
