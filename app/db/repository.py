"""Read/write helpers for request logs, tool calls and financial metrics.

Logging failures never break the API response."""

import logging

from sqlalchemy import select

from app.db.models import FinancialMetric, RequestLog, ToolCall
from app.db.session import session_scope

logger = logging.getLogger(__name__)


def log_request(
    *,
    request_id: str,
    endpoint: str,
    request_payload: dict,
    status: str,
    response_payload: dict | None = None,
    error_message: str | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
) -> None:
    try:
        with session_scope() as s:
            s.add(
                RequestLog(
                    request_id=request_id,
                    endpoint=endpoint,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    status=status,
                    error_message=error_message,
                    model=model,
                    latency_ms=latency_ms,
                )
            )
    except Exception:  # pragma: no cover - we still want the API to answer
        logger.exception("Failed to write request log %s to the database", request_id)


def get_request_log(request_id: str) -> RequestLog | None:
    with session_scope() as s:
        return s.scalars(select(RequestLog).where(RequestLog.request_id == request_id)).first()


def log_tool_calls(request_id: str, calls: list[dict]) -> None:
    """Store the agent's tool-call trace for a request."""
    if not calls:
        return
    try:
        with session_scope() as s:
            for i, c in enumerate(calls, start=1):
                s.add(
                    ToolCall(
                        request_id=request_id,
                        sequence=i,
                        tool_name=c.get("tool", "?"),
                        tool_input=(c.get("input") or "")[:4000],
                        output_chars=c.get("output_chars"),
                        duration_ms=c.get("duration_ms"),
                        error=c.get("error"),
                    )
                )
    except Exception:  # pragma: no cover
        logger.exception("Failed to write tool calls for %s", request_id)


def get_tool_calls(request_id: str) -> list[ToolCall]:
    with session_scope() as s:
        return list(
            s.scalars(select(ToolCall).where(ToolCall.request_id == request_id).order_by(ToolCall.sequence))
        )


def upsert_financial_metrics(rows: list[dict]) -> None:
    """Insert or update one row per quarter."""
    with session_scope() as s:
        for row in rows:
            rec = s.scalars(select(FinancialMetric).where(FinancialMetric.quarter == row["quarter"])).first()
            if rec is None:
                rec = FinancialMetric(quarter=row["quarter"])
                s.add(rec)
            rec.revenue_cr = row.get("revenue_cr")
            rec.operating_income_cr = row.get("operating_income_cr")
            rec.operating_margin_pct = row.get("operating_margin_pct")
            rec.net_profit_cr = row.get("net_profit_cr")
            rec.eps_inr = row.get("eps_inr")
            rec.metrics = row
            rec.source = (row.get("source") or "")[:255] or None
            rec.extraction_method = row.get("extraction_method")
            rec.verified = bool(row.get("verified"))
