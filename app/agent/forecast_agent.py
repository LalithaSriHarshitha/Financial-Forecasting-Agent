"""The LangChain tool-calling agent that chains both tools into a forecast."""

import logging
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from app.agent.progress import ProgressLogger
from app.agent.prompts import MASTER_PROMPT
from app.llm import get_chat_model
from app.schemas.forecast import ForecastResult
from app.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Upper bound on reasoning/tool steps per request (each tool call = 2 graph steps).
RECURSION_LIMIT = 40


@lru_cache
def build_agent():
    return create_agent(
        model=get_chat_model(),
        tools=ALL_TOOLS,
        system_prompt=MASTER_PROMPT,
        # ToolStrategy: the model "calls" the schema as a final tool; invalid output is
        # sent back to the model with the validation error so it can retry.
        response_format=ToolStrategy(ForecastResult),
    )


def run_forecast(task: str) -> tuple[ForecastResult, list[str]]:
    """Run the agent on a task. Returns the structured result and the tool-call trace."""
    agent = build_agent()
    state = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": RECURSION_LIMIT, "callbacks": [ProgressLogger()]},
    )
    trace = [
        f"{call['name']}({call.get('args', {})})"
        for msg in state["messages"]
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    logger.info("Agent tool calls: %s", trace)
    result = state.get("structured_response")
    if result is None:
        raise RuntimeError("The agent finished without producing a structured forecast.")
    if isinstance(result, dict):
        result = ForecastResult.model_validate(result)
    return result, trace
