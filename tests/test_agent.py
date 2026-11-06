"""Runs the real LangChain agent graph with a scripted fake LLM (no network).

Proves the chain works: the agent calls both tools, returns an AgentForecast through the
structured-output tool, and the grounding step attaches validated metrics and removes
quotes that are not in the transcripts.
"""

import importlib
import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.agent import forecast_agent
from app.schemas.forecast import AgentForecast, ForecastResult

# Import the modules (the package re-exports tool objects under the same names).
financial_extractor = importlib.import_module("app.tools.financial_extractor")
qualitative_analysis = importlib.import_module("app.tools.qualitative_analysis")
index = importlib.import_module("app.ingestion.index")


class ScriptedToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeRetriever:
    docs = [
        Document(
            page_content="[Samir Seksaria (management, qa)]: We remain confident about the medium-term margin trajectory.",
            metadata={"quarter": "Q2 FY27", "speaker": "Samir Seksaria", "role": "management", "page": 9},
        )
    ]


@pytest.fixture
def scripted_agent(monkeypatch, sample_metrics, sample_result):
    monkeypatch.setattr(financial_extractor, "get_financial_metrics", lambda quarters=None: sample_metrics)
    monkeypatch.setattr(
        qualitative_analysis, "analyze",
        lambda queries, quarter=None, speaker_role=None: {"queries": queries, "insights": {"sentiment": "neutral"}},
    )
    monkeypatch.setattr(index, "get_hybrid_retriever", lambda: FakeRetriever())

    def make(final_args: dict):
        script = [
            AIMessage(content="", tool_calls=[{"name": "financial_data_extractor", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "qualitative_analysis",
                                               "args": {"queries": ["demand outlook", "risks"]}, "id": "c2"}]),
            AIMessage(content="", tool_calls=[{"name": "AgentForecast", "args": final_args, "id": "c3"}]),
        ]
        monkeypatch.setattr(forecast_agent, "get_chat_model", lambda: ScriptedToolModel(responses=script))
        forecast_agent.build_agent.cache_clear()

    yield make
    forecast_agent.build_agent.cache_clear()


def _agent_args(sample_result: ForecastResult, **changes) -> dict:
    args = json.loads(AgentForecast(**sample_result.model_dump()).model_dump_json())
    args.update(changes)
    return args


def test_agent_chains_tools_and_grounds_output(scripted_agent, caplog, sample_result):
    caplog.set_level("INFO", logger="agent.progress")
    args = _agent_args(sample_result)
    args["management_outlook"]["key_quotes"] = [
        {"text": "We remain confident about the medium-term margin trajectory.", "quarter": "Q1 FY27"},
        {"text": "We will grow revenue 15% next quarter with record margins.", "quarter": "Q2 FY27"},
    ]
    scripted_agent(args)

    run = forecast_agent.run_forecast("Forecast next quarter for TCS")

    assert isinstance(run.result, ForecastResult)
    assert run.result.forecast_period == "Q3 FY27"
    assert [t.split("(")[0] for t in run.trace] == ["financial_data_extractor", "qualitative_analysis"]
    # numbers come from the extractor, not the LLM
    q2 = next(m for m in run.result.quarterly_metrics if m.quarter == "Q2 FY27")
    assert q2.revenue_cr == 73188 and q2.revenue_growth_qoq_pct == 1.26
    # the real quote is kept with corrected quarter + speaker; the invented one is removed
    quotes = run.result.management_outlook.key_quotes
    assert len(quotes) == 1 and quotes[0].quarter == "Q2 FY27" and quotes[0].speaker == "Samir Seksaria"
    assert len(run.grounding["quotes_removed_as_unsupported"]) == 1
    assert any("removed" in g for g in run.result.data_gaps)
    # tool calls recorded for MySQL + live progress logged
    assert [c["tool"] for c in run.tool_calls] == ["financial_data_extractor", "qualitative_analysis"]
    assert all(c["duration_ms"] >= 0 for c in run.tool_calls)
    assert "LLM call #3 started" in caplog.text and "Tool qualitative_analysis started" in caplog.text


def test_wrong_forecast_period_is_corrected(scripted_agent, sample_result):
    scripted_agent(_agent_args(sample_result, forecast_period="Q4 FY27"))
    run = forecast_agent.run_forecast("Forecast next quarter for TCS")
    assert run.result.forecast_period == "Q3 FY27"
    assert run.grounding["forecast_period_corrected"] == {"from": "Q4 FY27", "to": "Q3 FY27"}


def test_market_snapshot_attached_and_gap_reported(scripted_agent, sample_result, monkeypatch):
    md = importlib.import_module("app.tools.market_data")
    monkeypatch.setattr(md.get_settings(), "market_data_enabled", True)
    monkeypatch.setattr(md, "get_market_snapshot", lambda force=False: {"available": False, "error": "offline"})
    scripted_agent(_agent_args(sample_result, market_context="Shares trade near lows."))
    run = forecast_agent.run_forecast("Forecast next quarter for TCS")
    assert run.result.market_snapshot == {"available": False, "error": "offline"}
    assert run.result.market_context is None
    assert any("market data unavailable" in g.lower() for g in run.result.data_gaps)
    assert run.usage["llm_calls"] == 3 and run.usage["tool_calls"] == 2
