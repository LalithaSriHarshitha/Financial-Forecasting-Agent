"""Request/response models for the /forecast endpoint.

`ForecastResult` is also the structured-output schema the agent must fill in,
so every field description here doubles as an instruction to the LLM.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_TASK = (
    "Analyze the financial reports and transcripts for the last three quarters and provide "
    "a qualitative forecast for the upcoming quarter. Your forecast must identify key financial "
    "trends (e.g., revenue growth, margin pressure), summarize management's stated outlook, and "
    "highlight any significant risks or opportunities mentioned."
)


class ForecastRequest(BaseModel):
    task: str = Field(
        default=DEFAULT_TASK,
        min_length=10,
        max_length=4000,
        description="The analytical task for the agent.",
    )


class QuarterSnapshot(BaseModel):
    """Validated figures for one quarter, filled by code from the FinancialDataExtractorTool
    (never written by the LLM)."""

    quarter: str
    revenue_cr: float | None = None
    revenue_growth_qoq_pct: float | None = None
    revenue_growth_yoy_pct: float | None = None
    cc_growth_qoq_pct: float | None = None
    operating_margin_pct: float | None = None
    employee_cost_pct: float | None = None
    net_profit_cr: float | None = None
    net_margin_pct: float | None = None
    eps_inr: float | None = None
    order_book_tcv_usd_bn: float | None = None
    headcount: int | None = None
    attrition_ltm_pct: float | None = None
    verified: bool = False
    source: str | None = None


class FinancialTrend(BaseModel):
    metric: str = Field(description="e.g. 'Revenue growth', 'Operating margin'.")
    direction: Literal["improving", "stable", "declining", "mixed"]
    evidence: str = Field(description="Numbers from the tool output that support the direction.")


class Quote(BaseModel):
    text: str = Field(description="Management quote copied word for word from the tool's verified statements.")
    quarter: str
    speaker: str | None = None


class ManagementOutlook(BaseModel):
    summary: str
    sentiment: Literal["positive", "cautiously_optimistic", "neutral", "cautious", "negative"]
    key_quotes: list[Quote] = Field(default_factory=list)


class Insight(BaseModel):
    title: str
    detail: str
    source: str = Field(description="Where this came from, e.g. 'Q2 FY27 earnings call' or 'Q1 FY27 fact sheet'.")


class Outlook(BaseModel):
    revenue_outlook: str = Field(description="Expected direction of revenue next quarter, with reasoning.")
    margin_outlook: str = Field(description="Expected direction of operating margin next quarter, with reasoning.")
    overall_outlook: str = Field(description="2-4 sentence qualitative forecast for the next quarter.")
    confidence: Literal["low", "medium", "high"]


class AgentForecast(BaseModel):
    """What the agent writes. Numbers per quarter are attached afterwards by code."""

    company: str = "Tata Consultancy Services (TCS)"
    forecast_period: str = Field(description="The quarter being forecast, e.g. 'Q3 FY27'.")
    quarters_analyzed: list[str] = Field(description="The (up to three) latest quarters analysed, oldest first.")
    financial_trends: list[FinancialTrend]
    management_outlook: ManagementOutlook
    risks: list[Insight]
    opportunities: list[Insight]
    forecast: Outlook
    data_gaps: list[str] = Field(
        default_factory=list, description="Anything missing or uncertain in the source data."
    )


class ForecastResult(AgentForecast):
    """The API's result: the agent's analysis plus validated per-quarter metrics."""

    quarterly_metrics: list[QuarterSnapshot] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    request_id: str
    status: Literal["success"]
    model: str
    generated_at: datetime
    latency_ms: int
    tool_calls: list[str] = Field(default_factory=list, description="Tools the agent called, in order.")
    grounding: dict = Field(default_factory=dict, description="Results of the post-generation checks.")
    result: ForecastResult
