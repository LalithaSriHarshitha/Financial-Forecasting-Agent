"""Schemas produced by the FinancialDataExtractorTool and QualitativeAnalysisTool."""

from typing import Literal

from pydantic import BaseModel, Field


class QuarterMetrics(BaseModel):
    """Key consolidated metrics for one quarter, in INR crore unless stated."""

    quarter: str = Field(description="Fiscal quarter label, e.g. 'Q2 FY27'.")
    revenue_cr: float | None = Field(None, description="Total revenue, INR crore.")
    operating_income_cr: float | None = Field(
        None, description="Operating income / profit before taxes and other income, INR crore."
    )
    operating_margin_pct: float | None = Field(
        None, description="Operating margin as % of revenue."
    )
    net_profit_cr: float | None = Field(
        None, description="Net profit attributable to shareholders, INR crore."
    )
    net_margin_pct: float | None = Field(None, description="Net profit as % of revenue.")
    eps_inr: float | None = Field(None, description="Earnings per share, INR.")


class ExtractedStatement(BaseModel):
    """All quarter columns read from one income-statement table."""

    quarters: list[QuarterMetrics] = Field(
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
