"""Database tables."""

from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RequestLog(Base):
    """One row per API request: what came in, what went out, and how it went."""

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(100))
    request_payload: Mapped[dict] = mapped_column(JSON)
    response_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), index=True)  # success | error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class FinancialMetric(Base):
    """Latest extracted metrics per quarter (upserted by the FinancialDataExtractorTool)."""

    __tablename__ = "financial_metrics"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    quarter: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    revenue_cr: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_income_cr: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margin_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_profit_cr: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_inr: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON)  # every extracted field
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extraction_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class ToolCall(Base):
    """One row per tool call made by the agent while answering a request."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(100))
    tool_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
