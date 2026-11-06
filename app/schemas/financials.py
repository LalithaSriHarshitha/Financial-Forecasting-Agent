"""Schemas produced by the FinancialDataExtractorTool and QualitativeAnalysisTool."""

from typing import Literal

from pydantic import BaseModel, Field


class LLMQuarterRow(BaseModel):
    """Fallback schema the LLM fills when rules-based parsing fails (INR crore)."""

    quarter: str = Field(description="Fiscal quarter label, e.g. 'Q2 FY27'.")
    revenue_cr: float | None = Field(None, description="Total revenue, INR crore.")
    operating_income_cr: float | None = Field(
        None, description="Operating income / profit before taxes and other income, INR crore."
    )
    net_profit_cr: float | None = Field(
        None, description="Net profit attributable to shareholders, INR crore."
    )
    eps_inr: float | None = Field(None, description="Earnings per share, INR.")


class QuarterMetrics(BaseModel):
    """All metrics for one quarter. Money in INR crore unless stated; ratios in %."""

    quarter: str
    # Income statement (Ind AS, consolidated)
    revenue_cr: float | None = None
    employee_cost_cr: float | None = None
    employee_cost_pct: float | None = None
    operating_income_cr: float | None = None
    operating_margin_pct: float | None = None
    other_income_cr: float | None = None
    profit_before_tax_cr: float | None = None
    net_profit_cr: float | None = None
    net_margin_pct: float | None = None
    eps_inr: float | None = None
    # Operational signals (performance highlights page)
    usd_revenue_mn: float | None = None
    usd_growth_qoq_pct: float | None = None
    usd_growth_yoy_pct: float | None = None
    cc_growth_qoq_pct: float | None = None
    cc_growth_yoy_pct: float | None = None
    order_book_tcv_usd_bn: float | None = None
    headcount: int | None = None
    attrition_ltm_pct: float | None = None
    cfo_to_net_profit_pct: float | None = None
    # Provenance
    source: str | None = None
    extraction_method: str | None = None  # "rules" | "llm_fallback"
    verified: bool = False  # all accounting checks passed


class ExtractedStatement(BaseModel):
    """All quarter columns read from one income-statement table."""

    quarters: list[LLMQuarterRow] = Field(
        description="One entry per quarter column found in the table."
    )
    notes: str | None = Field(
        None, description="Caveats, e.g. 'excludes exceptional items'."
    )


class Theme(BaseModel):
    theme: str = Field(description="Short name of the theme, e.g. 'AI-led deal wins'.")
    summary: str = Field(description="What management said about it, 1-2 sentences.")
    quarters: list[str] = Field(default_factory=list, description="Quarters where it appears.")


class ForwardLookingStatement(BaseModel):
    statement: str = Field(description="Verbatim or near-verbatim quote from the transcript.")
    quarter: str
    speaker: str | None = None


class QualitativeInsights(BaseModel):
    themes: list[Theme] = Field(default_factory=list)
    sentiment: Literal["positive", "cautiously_optimistic", "neutral", "cautious", "negative"]
    sentiment_rationale: str
    forward_looking_statements: list[ForwardLookingStatement] = Field(default_factory=list)
    risks_mentioned: list[str] = Field(default_factory=list)
    opportunities_mentioned: list[str] = Field(default_factory=list)
