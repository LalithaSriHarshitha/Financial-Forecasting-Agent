"""One-command end-to-end run:  python -m app.demo

1. checks documents and builds the transcript index if it is missing,
2. creates the MySQL database and tables if needed,
3. sends the assignment's example task to the real POST /forecast endpoint (in-process),
4. confirms the request/response row was logged to MySQL,
5. saves the JSON to sample_output.json and prints a short summary.
"""

import json
import sys
import time

from fastapi.testclient import TestClient

from app import __version__
from app.config import PROJECT_ROOT, get_settings
from app.logging_config import setup_logging


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    from app.ingestion.manifest import load_manifest

    missing = [d.filename for d in load_manifest() if not d.exists]
    if missing:
        print("Missing documents in data/raw/:", ", ".join(missing))
        print("Run: python -m app.ingestion   (download any blocked files manually, see README)")
        return 1
    if not settings.chroma_path.exists():
        print("Building the transcript index first ...")
        from app.ingestion.index import build_index

        build_index()

    from app.db.repository import get_request_log
    from app.main import app

    print(f"Running the forecast agent with {settings.chat_model_name} (a few minutes) ...")
    started = time.perf_counter()
    with TestClient(app) as client:  # startup creates the database and tables
        resp = client.post("/forecast", json={})
    elapsed = time.perf_counter() - started

    body = resp.json()
    if resp.status_code != 200:
        print(f"FAILED (HTTP {resp.status_code}):", json.dumps(body, indent=2))
        return 1

    out = PROJECT_ROOT / "sample_output.json"
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    row = get_request_log(body["request_id"])
    result = body["result"]

    print(f"\n================ END-TO-END RUN (v{__version__}) ================")
    print(f"HTTP status          : {resp.status_code} in {elapsed:.0f}s")
    print(f"Request id           : {body['request_id']}")
    print(f"Logged to MySQL      : {'yes, status=' + row.status if row else 'NO'}")
    print(f"Tool calls           : {len(body['tool_calls'])}")
    for call in body["tool_calls"]:
        print("   -", call[:110])
    print(f"Forecast period      : {result['forecast_period']}")
    print(f"Quarters analysed    : {', '.join(result['quarters_analyzed'])}")
    for m in result["quarterly_metrics"]:
        print(f"   {m['quarter']}: revenue {m['revenue_cr']:,.0f} cr (QoQ {m['revenue_growth_qoq_pct']}%, "
              f"CC {m['cc_growth_qoq_pct']}%), op margin {m['operating_margin_pct']}%, "
              f"net profit {m['net_profit_cr']:,.0f} cr, TCV ${m['order_book_tcv_usd_bn']} bn, "
              f"attrition {m['attrition_ltm_pct']}%  [{'verified' if m['verified'] else 'UNVERIFIED'}]")
    print(f"Management sentiment : {result['management_outlook']['sentiment']}")
    g = body.get("grounding", {})
    print(f"Grounding            : accounting checks {g.get('accounting_checks')}, "
          f"quotes verified {g.get('quotes_verified')}, removed {len(g.get('quotes_removed_as_unsupported') or [])}")
    print(f"Risks / opportunities: {len(result['risks'])} / {len(result['opportunities'])}")
    print(f"Outlook              : {result['forecast']['overall_outlook']}")
    print(f"Direction            : revenue {result['forecast']['revenue_direction']}, "
          f"margin {result['forecast']['margin_direction']}")
    print(f"Confidence           : {result['forecast']['confidence']}")
    mk = result.get("market_snapshot") or {}
    if mk.get("available"):
        print(f"Market               : {mk['symbol']} INR {mk['price']} ({mk['day_change_pct']}% today), "
              f"{mk['pct_below_52w_high']}% below 52w high, P/E (TTM) {mk['pe_ttm']}")
    elif mk:
        print(f"Market               : unavailable ({mk.get('error')})")
    u = body.get("usage", {})
    print(f"Usage                : {u.get('llm_calls')} LLM calls, {u.get('input_tokens')} input / "
          f"{u.get('output_tokens')} output tokens")
    print(f"\nFull JSON saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
