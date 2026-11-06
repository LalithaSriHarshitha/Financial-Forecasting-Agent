"""Callback handler that logs every LLM call and tool call with timings.

Local models can take minutes per call; this makes progress visible in the console
(and in the API logs) instead of a silent wait.
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent.progress")


class ProgressLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        self.started: dict[UUID, float] = {}
        self.llm_calls = 0

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **kw: Any) -> None:
        self.llm_calls += 1
        chars = sum(len(str(getattr(m, "content", m))) for batch in messages for m in batch)
        self.started[run_id] = time.perf_counter()
        logger.info("LLM call #%d started (~%d input tokens) ...", self.llm_calls, chars // 4)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kw: Any) -> None:
        t0 = self.started.pop(run_id, None)
        if t0 is not None:
            logger.info("LLM call finished in %.0fs", time.perf_counter() - t0)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        self.started.pop(run_id, None)
        logger.warning("LLM call failed: %s", str(error)[:200])

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kw: Any) -> None:
        self.started[run_id] = time.perf_counter()
        logger.info("Tool %s started with %s", (serialized or {}).get("name", "?"), input_str[:150])

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw: Any) -> None:
        t0 = self.started.pop(run_id, None)
        if t0 is not None:
            logger.info("Tool finished in %.0fs", time.perf_counter() - t0)
