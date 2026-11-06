import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests use a throwaway SQLite database and a dummy key; no MySQL or Gemini needed.
os.environ["DATABASE_URL"] = f"sqlite:///{ROOT / 'tests' / '.test.db'}"
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ["MARKET_DATA_ENABLED"] = "false"  # no network in tests; market tests mock the quote
os.environ["LLM_PROVIDER"] = "gemini"


@pytest.fixture(scope="session", autouse=True)
def _db():
    from app.db.session import init_db

    db = ROOT / "tests" / ".test.db"
    db.unlink(missing_ok=True)
    init_db()
    yield
    from app.db.session import get_engine

    get_engine().dispose()
    db.unlink(missing_ok=True)


@pytest.fixture
def sample_metrics():
    return {
        "unit": "INR crore",
        "metrics": [
            {"quarter": "Q1 FY27", "revenue_cr": 72275, "operating_income_cr": 17317,
             "operating_margin_pct": 23.96, "net_profit_cr": 13849, "net_margin_pct": 19.16,
             "eps_inr": 38.28},
            {"quarter": "Q2 FY27", "revenue_cr": 73188, "operating_income_cr": 17553,
             "operating_margin_pct": 23.98, "net_profit_cr": 13884, "net_margin_pct": 18.97,
             "eps_inr": 38.37, "verified": True, "source": "TCS Q2 FY27 Fact Sheet, page 21"},
        ],
        "trends": [{"quarter": "Q2 FY27", "revenue_growth_qoq_pct": 1.26}],
        "validation": {"checks_passed": 42, "checks_total": 42, "methods": {}, "restatements": []},
        "errors": [],
    }


@pytest.fixture
def sample_result():
    from app.schemas.forecast import ForecastResult

    return ForecastResult.model_validate(
        {
            "forecast_period": "Q3 FY27",
            "quarters_analyzed": ["Q1 FY27", "Q2 FY27"],
            "quarterly_metrics": [
                {"quarter": "Q2 FY27", "revenue_cr": 73188, "operating_margin_pct": 23.98,
                 "net_profit_cr": 13884, "eps_inr": 38.37}
            ],
            "financial_trends": [
                {"metric": "Revenue growth", "direction": "improving",
                 "evidence": "Revenue up 1.26% QoQ to INR 73,188 crore"}
            ],
            "management_outlook": {
                "summary": "Management expects steady demand.",
                "sentiment": "cautiously_optimistic",
                "key_quotes": [{"text": "We remain confident.", "quarter": "Q2 FY27"}],
            },
            "risks": [{"title": "Discretionary spend", "detail": "Clients cautious",
                       "source": "Q2 FY27 earnings call"}],
            "opportunities": [{"title": "AI services", "detail": "Growing AI deals",
                               "source": "Q2 FY27 earnings call"}],
            "forecast": {
                "revenue_direction": "growth", "margin_direction": "stable",
                "revenue_outlook": "Modest growth", "margin_outlook": "Stable",
                "overall_outlook": "Steady quarter expected.", "confidence": "medium",
            },
        }
    )
