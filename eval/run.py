"""Evaluation suite:  python -m eval.run [--consistency N] [--output sample_output.json]

1. Extraction accuracy   - every field vs eval/ground_truth.json (no LLM).
2. Retrieval quality     - hit@k on hand-written questions for hybrid vs vector-only vs
                           BM25-only search (embeddings only, no chat LLM).
3. Output faithfulness   - deterministic checks on a saved forecast: metrics match ground truth,
                           every quote is in the transcripts, every risk/opportunity has a
                           source, forecast period is right, no investment advice.
4. Consistency (opt-in)  - runs the full agent N times and measures agreement of the forecast
                           directions and confidence (costs ~5 LLM requests per run).

Writes eval/report.md and eval/report.json.
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.logging_config import setup_logging

EVAL_DIR = Path(__file__).parent
QUESTIONS = EVAL_DIR / "retrieval_questions.json"
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
# Advice phrases, not bare words: client names such as "Best Buy" must not trigger it.
ADVICE_WORDS = re.compile(
    r"\b(?:(?:strong\s+)?(?:buy|sell|hold|accumulate)\s+(?:rating|recommendation|call)|"
    r"(?:buy|sell)\s+(?:the\s+)?(?:stock|shares)|recommend(?:ed|s)?\s+(?:buying|selling|holding)|"
    r"price\s+target|target\s+price|overweight|underweight)\b",
    re.I,
)


# ---------------- 1. Extraction ----------------

def eval_extraction() -> dict:
    from eval.extraction import run

    r = run()
    return {"matched": r["matched"], "total": r["total"], "accuracy_pct": r["accuracy_pct"],
            "mismatches": [x for x in r["rows"] if not x["match"]]}


# ---------------- 2. Retrieval ----------------

def _hit(docs, expect: list[str]) -> bool:
    return any(t.lower() in d.page_content.lower() for d in docs for t in expect)


def eval_retrieval() -> dict:
    from app.ingestion.index import get_hybrid_retriever, tokenize

    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    k, questions = spec["k"], spec["questions"]
    r = get_hybrid_retriever()
    vectors = r.store.embeddings.embed_documents([q["question"] for q in questions])
    rows, totals = [], Counter()
    for q, vec in zip(questions, vectors):
        quarter = q.get("quarter")
        flt = {"quarter": quarter} if quarter else None
        vector_docs = r.store.similarity_search_by_vector(vec, k=k, filter=flt)
        scores = r.bm25.get_scores(tokenize(q["question"]))
        ranked = sorted(range(len(r.docs)), key=lambda i: scores[i], reverse=True)
        bm25_docs = [r.docs[i] for i in ranked if not quarter or r.docs[i].metadata.get("quarter") == quarter][:k]
        hybrid_docs = r.search(q["question"], vec, k=k, quarter=quarter)
        row = {"id": q["id"], "quarter": quarter,
               "hybrid": _hit(hybrid_docs, q["expect_any"]),
               "vector_only": _hit(vector_docs, q["expect_any"]),
               "bm25_only": _hit(bm25_docs, q["expect_any"])}
        totals.update({m: row[m] for m in ("hybrid", "vector_only", "bm25_only")})
        rows.append(row)
    n = len(rows)
    return {"k": k, "questions": n,
            "hit_rate_pct": {m: round(totals[m] / n * 100, 1) for m in ("hybrid", "vector_only", "bm25_only")},
            "rows": rows}


# ---------------- 3. Faithfulness of a saved forecast ----------------

def eval_faithfulness(output_path: Path) -> dict:
    from app.ingestion.index import get_hybrid_retriever
    from app.tools.grounding import verify_quote
    from app.tools.quarters import next_quarter, normalize_quarter, quarter_sort_key

    if not output_path.exists():
        return {"skipped": f"{output_path.name} not found (run python -m app.demo first)"}
    body = json.loads(output_path.read_text(encoding="utf-8"))
    result = body.get("result", body)
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["quarters"]
    checks = []

    # metrics vs ground truth
    for m in result.get("quarterly_metrics", []):
        gt = truth.get(m["quarter"])
        if not gt:
            continue
        for f in ("revenue_cr", "net_profit_cr", "eps_inr", "order_book_tcv_usd_bn", "attrition_ltm_pct"):
            if f in gt and m.get(f) is not None:
                checks.append({"check": f"{m['quarter']} {f} matches ground truth",
                               "passed": abs(float(m[f]) - float(gt[f])) <= 0.011})

    # quotes found in transcripts
    corpus = get_hybrid_retriever().docs
    for q in result.get("management_outlook", {}).get("key_quotes", []):
        ok, _ = verify_quote(q["text"], corpus, normalize_quarter(q.get("quarter", "")))
        checks.append({"check": f"quote in transcripts: {q['text'][:60]}...", "passed": ok})

    # every risk / opportunity cites a quarter
    for kind in ("risks", "opportunities"):
        for item in result.get(kind, []):
            checks.append({"check": f"{kind[:-1]} '{item['title'][:40]}' cites a source quarter",
                           "passed": bool(re.search(r"Q[1-4]\s*FY\d{2}", item.get("source", "")))})

    # forecast period
    analysed = sorted(result.get("quarters_analyzed", []), key=quarter_sort_key)
    if analysed:
        checks.append({"check": "forecast period is the quarter after the latest analysed",
                       "passed": normalize_quarter(result.get("forecast_period", "")) == next_quarter(analysed[-1])})

    # no investment advice
    text = json.dumps({k: result.get(k) for k in ("forecast", "market_context", "risks", "opportunities")})
    checks.append({"check": "no investment advice (buy/sell/price target)", "passed": not ADVICE_WORDS.search(text)})

    passed = sum(c["passed"] for c in checks)
    return {"file": output_path.name, "passed": passed, "total": len(checks),
            "score_pct": round(passed / len(checks) * 100, 1) if checks else None,
            "failed": [c["check"] for c in checks if not c["passed"]], "checks": checks}


# ---------------- 4. Consistency ----------------

def eval_consistency(runs: int) -> dict:
    from app.agent.forecast_agent import run_forecast
    from app.schemas.forecast import DEFAULT_TASK

    outcomes, errors = [], []
    for i in range(runs):
        try:
            res = run_forecast(DEFAULT_TASK).result.forecast
            outcomes.append((res.revenue_direction, res.margin_direction, res.confidence))
            print(f"  run {i + 1}/{runs}: revenue {res.revenue_direction}, margin {res.margin_direction}, "
                  f"confidence {res.confidence}")
        except Exception as exc:
            errors.append(f"run {i + 1}: {type(exc).__name__}: {str(exc)[:150]}")
    if not outcomes:
        return {"runs": runs, "completed": 0, "errors": errors}

    def agreement(idx):
        values = [o[idx] for o in outcomes]
        top, count = Counter(values).most_common(1)[0]
        return {"mode": top, "agreement_pct": round(count / len(values) * 100, 1), "values": values}

    return {"runs": runs, "completed": len(outcomes), "errors": errors,
            "revenue_direction": agreement(0), "margin_direction": agreement(1), "confidence": agreement(2)}


# ---------------- report ----------------

def write_report(report: dict) -> Path:
    lines = [f"# Evaluation report", "", f"Generated {report['generated_at']} with "
             f"`{report['provider']}` / `{report['model']}`.", ""]
    ex = report["extraction"]
    lines += ["## 1. Extraction accuracy", "",
              f"**{ex['matched']}/{ex['total']} = {ex['accuracy_pct']}%** of ground-truth values matched "
              "(rules-based parser, no LLM).", ""]
    if ex["mismatches"]:
        lines += ["| Quarter | Field | Expected | Got |", "|---|---|---|---|"]
        lines += [f"| {m['quarter']} | {m['field']} | {m['expected']} | {m['actual']} |" for m in ex["mismatches"]]
        lines.append("")
    rt = report.get("retrieval")
    if rt and "rows" in rt:
        h = rt["hit_rate_pct"]
        lines += ["## 2. Retrieval quality", "",
                  f"Hit@{rt['k']} on {rt['questions']} hand-written questions (paraphrased, no key terms):", "",
                  "| Method | Hit rate |", "|---|---|",
                  f"| **Hybrid (BM25 + vector, RRF)** | **{h['hybrid']}%** |",
                  f"| Vector only | {h['vector_only']}% |", f"| BM25 only | {h['bm25_only']}% |", "",
                  "| Question | Quarter | Hybrid | Vector | BM25 |", "|---|---|---|---|---|"]
        mark = lambda b: "yes" if b else "no"  # noqa: E731
        lines += [f"| {r['id']} | {r['quarter'] or 'all'} | {mark(r['hybrid'])} | {mark(r['vector_only'])} | "
                  f"{mark(r['bm25_only'])} |" for r in rt["rows"]]
        lines.append("")
    elif rt:
        lines += ["## 2. Retrieval quality", "", f"Skipped: {rt.get('skipped') or rt.get('error')}", ""]
    fa = report["faithfulness"]
    lines += ["## 3. Output faithfulness", ""]
    if "checks" in fa:
        lines += [f"**{fa['passed']}/{fa['total']} = {fa['score_pct']}%** of checks passed on `{fa['file']}`.", ""]
        lines += [f"- Failed: {c}" for c in fa["failed"]] or ["- All checks passed."]
        lines.append("")
    else:
        lines += [f"Skipped: {fa.get('skipped') or fa.get('error')}", ""]
    co = report.get("consistency")
    lines += ["## 4. Consistency", ""]
    if co and co.get("completed"):
        lines += [f"{co['completed']} of {co['runs']} runs completed.", "",
                  "| Field | Most common | Agreement | Values |", "|---|---|---|---|"]
        for f in ("revenue_direction", "margin_direction", "confidence"):
            lines.append(f"| {f} | {co[f]['mode']} | {co[f]['agreement_pct']}% | {', '.join(co[f]['values'])} |")
        lines.append("")
    else:
        lines += ["Not run. Use `python -m eval.run --consistency 3` (about 5 LLM requests per run).", ""]
    out = EVAL_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (EVAL_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction, retrieval, faithfulness, consistency.")
    parser.add_argument("--consistency", type=int, default=0, metavar="N", help="run the agent N times")
    parser.add_argument("--output", default="sample_output.json", help="saved forecast to check")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    s = get_settings()
    setup_logging("WARNING")

    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "provider": s.llm_provider, "model": s.chat_model_name}
    print("1/4 Extraction accuracy ...")
    report["extraction"] = eval_extraction()
    print(f"    {report['extraction']['matched']}/{report['extraction']['total']}")
    print("2/4 Retrieval quality ...")
    try:
        report["retrieval"] = {"skipped": "--skip-retrieval"} if args.skip_retrieval else eval_retrieval()
        if "hit_rate_pct" in report["retrieval"]:
            print(f"    hit rate {report['retrieval']['hit_rate_pct']}")
    except Exception as exc:
        report["retrieval"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("    error:", exc)
    print("3/4 Output faithfulness ...")
    try:
        report["faithfulness"] = eval_faithfulness(PROJECT_ROOT / args.output)
        print(f"    {report['faithfulness'].get('passed')}/{report['faithfulness'].get('total')}")
    except Exception as exc:
        report["faithfulness"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("    error:", exc)
    print("4/4 Consistency ...")
    report["consistency"] = eval_consistency(args.consistency) if args.consistency > 0 else None
    out = write_report(report)
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
