"""Runs the real LangChain agent graph with a scripted fake LLM (no network).

Proves the chain works: the agent calls both tools, then returns a ForecastResult
through the structured-output tool.
"""

import importlib
import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.agent import forecast_agent
from app.schemas.forecast import ForecastResult

# Import the modules (the package re-exports tool objects under the same names).
financial_extractor = importlib.import_module("app.tools.financial_extractor")
qualitative_analysis = importlib.import_module("app.tools.qualitative_analysis")


class ScriptedToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_chains_tools_into_structured_forecast(monkeypatch, sample_metrics, sample_result):
    monkeypatch.setattr(financial_extractor, "get_financial_metrics", lambda quarters=None: sample_metrics)
    monkeypatch.setattr(
        qualitative_analysis, "analyze",
        lambda queries, quarter=None: {"queries": queries, "insights": {"sentiment": "cautiously_optimistic"}},
    )
    script = [
        AIMessage(content="", tool_calls=[{"name": "financial_data_extractor", "args": {}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "qualitative_analysis",
                                           "args": {"queries": ["demand outlook", "risks"]}, "id": "c2"}]),
        AIMessage(content="", tool_calls=[{"name": "ForecastResult",
                                           "args": json.loads(sample_result.model_dump_json()), "id": "c3"}]),
    ]
    monkeypatch.setattr(forecast_agent, "get_chat_model", lambda: ScriptedToolModel(responses=script))
    forecast_agent.build_agent.cache_clear()

    result, trace = forecast_agent.run_forecast("Forecast next quarter for TCS")

    assert isinstance(result, ForecastResult)
    assert result.forecast_period == "Q3 FY27"
    assert trace[0].startswith("financial_data_extractor")
    assert trace[1].startswith("qualitative_analysis")
    forecast_agent.build_agent.cache_clear()
