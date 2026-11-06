"""Extraction accuracy against hand-typed ground truth:  python -m eval.extraction

Compares every field in eval/ground_truth.json with what the FinancialDataExtractorTool
returns and prints a per-field report. Exit code 0 only if every value matches.
"""

import json
import sys
from pathlib import Path

from app.tools.financial_extractor import get_financial_metrics

TRUTH = Path(__file__).with_name("ground_truth.json")


def run() -> dict:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["quarters"]
    extracted = {m["quarter"]: m for m in get_financial_metrics()["metrics"]}
    rows, ok = [], 0
    for quarter, expected in truth.items():
        got = extracted.get(quarter, {})
        for field, value in expected.items():
            if field == "source":
                continue
            actual = got.get(field)
            match = actual is not None and abs(float(actual) - float(value)) <= 0.011
            ok += match
            rows.append({"quarter": quarter, "field": field, "expected": value, "actual": actual, "match": match})
    return {"total": len(rows), "matched": ok, "accuracy_pct": round(ok / len(rows) * 100, 1), "rows": rows}


def main() -> int:
    report = run()
    for r in report["rows"]:
        mark = "OK " if r["match"] else "XX "
        print(f"{mark} {r['quarter']:8} {r['field']:24} expected {r['expected']:>12}  got {r['actual']}")
    print(f"\nExtraction accuracy: {report['matched']}/{report['total']} = {report['accuracy_pct']}%")
    return 0 if report["matched"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
