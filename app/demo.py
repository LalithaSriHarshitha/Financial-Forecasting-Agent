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

    print(f"Running the forecast agent with {settings.gemini_model} (about 1-3 minutes) ...")
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

    print("\n================ V1 END-TO-END RUN ================")
    print(f"HTTP status          : {resp.status_code} in {elapsed:.0f}s")
    print(f"Request id           : {body['request_id']}")
    print(f"Logged to MySQL      : {'yes, status=' + row.status if row else 'NO'}")
    print(f"Tool calls           : {len(body['tool_calls'])}")
    for call in body["tool_calls"]:
        print("   -", call[:110])
    print(f"Forecast period      : {result['forecast_period']}")
    print(f"Quarters analysed    : {', '.join(result['quarters_analyzed'])}")
    for m in result["quarterly_metrics"]:
        print(f"   {m['quarter']}: revenue {m['revenue_cr']} cr, op margin {m['operating_margin_pct']}%, "
              f"net profit {m['net_profit_cr']} cr, EPS {m['eps_inr']}")
    print(f"Management sentiment : {result['management_outlook']['sentiment']}")
    print(f"Risks / opportunities: {len(result['risks'])} / {len(result['opportunities'])}")
    print(f"Outlook              : {result['forecast']['overall_outlook']}")
    print(f"Confidence           : {result['forecast']['confidence']}")
    print(f"\nFull JSON saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
