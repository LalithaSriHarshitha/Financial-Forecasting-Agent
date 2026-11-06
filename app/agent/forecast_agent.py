"""The LangChain tool-calling agent that chains both tools into a forecast,
followed by deterministic grounding checks on what the LLM wrote."""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from app.agent.progress import ProgressLogger
from app.agent.prompts import MASTER_PROMPT
from app.llm import get_chat_model
from app.schemas.forecast import AgentForecast, ForecastResult, QuarterSnapshot
from app.tools import ALL_TOOLS
from app.tools.quarters import next_quarter, normalize_quarter, quarter_sort_key

logger = logging.getLogger(__name__)

# Upper bound on reasoning/tool steps per request (each tool call = 2 graph steps).
RECURSION_LIMIT = 40


@dataclass
class ForecastRun:
    result: ForecastResult
    trace: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)


@lru_cache
def build_agent():
    return create_agent(
        model=get_chat_model(),
        tools=ALL_TOOLS,
        system_prompt=MASTER_PROMPT,
        # ToolStrategy: the model "calls" the schema as a final tool; invalid output is
        # sent back to the model with the validation error so it can retry.
        response_format=ToolStrategy(AgentForecast),
    )


def _attach_metrics(out: AgentForecast, grounding: dict) -> list[QuarterSnapshot]:
    """Per-quarter numbers come from the validated extractor, never from the LLM's text."""
    from app.tools.financial_extractor import get_financial_metrics

    data = get_financial_metrics()
    metrics = {m["quarter"]: m for m in data["metrics"]}
    trends = {t["quarter"]: t for t in data["trends"]}
    quarters = [q for q in (normalize_quarter(x) for x in out.quarters_analyzed) if q in metrics]
    if not quarters:  # fall back to the three latest quarters with operational data
        quarters = [q for q in sorted(metrics, key=quarter_sort_key) if metrics[q].get("headcount")][-3:]
    out.quarters_analyzed = quarters
    expected = next_quarter(quarters[-1]) if quarters else None
    if expected and normalize_quarter(out.forecast_period) != expected:
        grounding["forecast_period_corrected"] = {"from": out.forecast_period, "to": expected}
        out.forecast_period = expected
    snapshots = []
    for q in quarters:
        m, t = metrics[q], trends.get(q, {})
        snapshots.append(
            QuarterSnapshot(
                **{k: m.get(k) for k in QuarterSnapshot.model_fields if k in m},
                revenue_growth_qoq_pct=t.get("revenue_growth_qoq_pct"),
                revenue_growth_yoy_pct=t.get("revenue_growth_yoy_pct"),
            )
        )
    grounding["metrics_from_validated_extractor"] = True
    grounding["quarters_verified"] = sum(s.verified for s in snapshots)
    v = data.get("validation") or {}
    grounding["accounting_checks"] = f"{v.get('checks_passed', 0)}/{v.get('checks_total', 0)} passed"
    return snapshots


def _verify_quotes(out: AgentForecast, grounding: dict) -> None:
    """Keep only key quotes that are found in the transcripts; fix their quarter/speaker."""
    from app.ingestion.index import get_hybrid_retriever
    from app.tools.grounding import verify_quote

    try:
        corpus = get_hybrid_retriever().docs
    except Exception as exc:
        grounding["quotes_checked"] = False
        logger.warning("Quote verification skipped: %s", exc)
        return
    kept, removed = [], []
    for quote in out.management_outlook.key_quotes:
        ok, src = verify_quote(quote.text, corpus, normalize_quarter(quote.quarter))
        if ok:
            quote.quarter = src.metadata["quarter"]
            if src.metadata.get("role") == "management":
                quote.speaker = src.metadata.get("speaker")
            kept.append(quote)
        else:
            removed.append(quote.text[:120])
    out.management_outlook.key_quotes = kept
    grounding["quotes_checked"] = True
    grounding["quotes_verified"] = len(kept)
    grounding["quotes_removed_as_unsupported"] = removed
    if removed:
        out.data_gaps.append(f"{len(removed)} quote(s) could not be found in the transcripts and were removed.")


def run_forecast(task: str, progress: ProgressLogger | None = None) -> ForecastRun:
    """Run the agent on a task, then ground its output. Pass `progress` to keep the
    tool-call records even if the run fails."""
    agent = build_agent()
    progress = progress or ProgressLogger()
    state = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": RECURSION_LIMIT, "callbacks": [progress]},
    )
    trace = [
        f"{call['name']}({call.get('args', {})})"
        for msg in state["messages"]
        for call in (getattr(msg, "tool_calls", None) or [])
        if call["name"] != AgentForecast.__name__  # the final structured answer, not a tool
    ]
    logger.info("Agent tool calls: %s", trace)
    out = state.get("structured_response")
    if out is None:
        raise RuntimeError("The agent finished without producing a structured forecast.")
    if isinstance(out, dict):
        out = AgentForecast.model_validate(out)

    grounding: dict = {}
    snapshots = _attach_metrics(out, grounding)
    _verify_quotes(out, grounding)
    result = ForecastResult(**out.model_dump(), quarterly_metrics=snapshots)
    return ForecastRun(result=result, trace=trace, tool_calls=progress.tool_calls, grounding=grounding)
