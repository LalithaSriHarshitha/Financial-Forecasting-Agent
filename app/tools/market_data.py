"""MarketDataTool (optional bonus): live TCS share-price context.

Price data comes from Yahoo Finance via yfinance (NSE symbol TCS.NS; quotes may be delayed
about 15 minutes). Valuation is computed from our own validated figures: trailing-twelve-month
EPS is the sum of the last four quarters' EPS from the FinancialDataExtractorTool, so
P/E = price / TTM EPS uses audited numbers rather than a third-party estimate.

Results are cached for a few minutes; if the quote cannot be fetched the tool returns
`available: false` and the forecast continues without it.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone

from langchain_core.tools import tool

from app.config import get_settings
from app.tools.quarters import previous_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

CACHE_SECONDS = 300
_cache: dict = {"at": 0.0, "data": None}
_lock = threading.Lock()


def fetch_quote(symbol: str) -> dict:
    """Raw quote fields from Yahoo Finance (network call)."""
    import yfinance as yf

    fi = yf.Ticker(symbol).fast_info
    quote = {k: fi.get(k) for k in (
        "currency", "lastPrice", "previousClose", "dayHigh", "dayLow", "yearHigh", "yearLow",
        "fiftyDayAverage", "twoHundredDayAverage", "yearChange", "marketCap",
    )}
    if not quote.get("lastPrice"):
        raise RuntimeError(f"no price returned for {symbol}")
    return quote


def ttm_eps() -> tuple[float | None, list[str]]:
    """Sum of EPS over the latest four consecutive quarters, from the validated extractor."""
    from app.tools.financial_extractor import get_financial_metrics

    metrics = {m["quarter"]: m for m in get_financial_metrics()["metrics"] if m.get("eps_inr") is not None}
    if not metrics:
        return None, []
    latest = max(metrics, key=quarter_sort_key)
    quarters = [latest] + [previous_quarter(latest, i) for i in range(1, 4)]
    if not all(q in metrics for q in quarters):
        return None, []
    return round(sum(metrics[q]["eps_inr"] for q in quarters), 2), list(reversed(quarters))


def _r(x, nd=2):
    return round(float(x), nd) if x is not None else None


def build_snapshot(quote: dict, symbol: str) -> dict:
    price, prev = quote["lastPrice"], quote.get("previousClose")
    eps, eps_quarters = ttm_eps()
    high = quote.get("yearHigh")
    return {
        "available": True,
        "symbol": symbol,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "currency": quote.get("currency") or "INR",
        "price": _r(price),
        "day_change_pct": _r((price - prev) / prev * 100) if prev else None,
        "fifty_two_week_high": _r(high),
        "fifty_two_week_low": _r(quote.get("yearLow")),
        "pct_below_52w_high": _r((high - price) / high * 100) if high else None,
        "fifty_day_average": _r(quote.get("fiftyDayAverage")),
        "two_hundred_day_average": _r(quote.get("twoHundredDayAverage")),
        "one_year_change_pct": _r(quote["yearChange"] * 100) if quote.get("yearChange") is not None else None,
        "market_cap_inr_cr": _r(quote["marketCap"] / 1e7, 0) if quote.get("marketCap") else None,
        "eps_ttm_inr": eps,
        "eps_ttm_quarters": eps_quarters,
        "pe_ttm": _r(price / eps) if eps else None,
        "source": "Yahoo Finance via yfinance (may be delayed ~15 min); EPS from validated TCS fact sheets",
    }


def get_market_snapshot(force: bool = False) -> dict:
    s = get_settings()
    if not s.market_data_enabled:
        return {"available": False, "error": "market data disabled (MARKET_DATA_ENABLED=false)"}
    with _lock:
        if not force and _cache["data"] and time.monotonic() - _cache["at"] < CACHE_SECONDS:
            return _cache["data"]
    try:
        data = build_snapshot(fetch_quote(s.market_symbol), s.market_symbol)
    except Exception as exc:
        logger.warning("Market data unavailable: %s", exc)
        return {"available": False, "symbol": s.market_symbol, "error": f"{type(exc).__name__}: {exc}"[:300]}
    with _lock:
        _cache.update(at=time.monotonic(), data=data)
    return data


@tool("market_data")
def market_data() -> str:
    """Current TCS share price context: price, day change, 52-week range, moving averages,
    1-year change, market cap, and trailing P/E computed from validated EPS.

    Use it as market context for the forecast (sentiment already priced in, valuation).
    If it returns available=false, continue without market data and note it in data_gaps.
    """
    return json.dumps(get_market_snapshot(), indent=2)
