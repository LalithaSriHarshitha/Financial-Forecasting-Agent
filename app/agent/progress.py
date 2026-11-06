"""Callback handler that logs every LLM call and tool call with timings,
and records the tool calls so they can be stored in MySQL (`tool_calls` table).

Local models can take minutes per call; this makes progress visible in the console
(and in the API logs) instead of a silent wait.
"""

import logging
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent.progress")


class ProgressLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        self.started: dict[UUID, float] = {}
        self.llm_calls = 0
        self.tool_calls: list[dict] = []
        self._open_tools: dict[UUID, dict] = {}
        self._lock = threading.Lock()  # tools may run in parallel threads

    # ---- LLM calls ----
    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID, **kw: Any) -> None:
        with self._lock:
            self.llm_calls += 1
            n = self.llm_calls
            self.started[run_id] = time.perf_counter()
        chars = sum(len(str(getattr(m, "content", m))) for batch in messages for m in batch)
        logger.info("LLM call #%d started (~%d input tokens) ...", n, chars // 4)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kw: Any) -> None:
        t0 = self.started.pop(run_id, None)
        if t0 is not None:
            logger.info("LLM call finished in %.0fs", time.perf_counter() - t0)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        self.started.pop(run_id, None)
        logger.warning("LLM call failed: %s", str(error)[:200])

    # ---- Tool calls ----
    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kw: Any) -> None:
        name = (serialized or {}).get("name") or kw.get("name") or "?"
        record = {"tool": name, "input": input_str, "t0": time.perf_counter()}
        with self._lock:
            self._open_tools[run_id] = record
            self.tool_calls.append(record)
        logger.info("Tool %s started with %s", name, input_str[:150])

    def _finish(self, run_id: UUID, output: Any = None, error: BaseException | None = None) -> None:
        with self._lock:
            record = self._open_tools.pop(run_id, None)
        if record is None:
            return
        record["duration_ms"] = int((time.perf_counter() - record.pop("t0")) * 1000)
        content = getattr(output, "content", output)
        record["output_chars"] = len(str(content)) if content is not None else None
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"[:1000]
        elif isinstance(content, str) and content.startswith('{"') and '"error"' in content[:200]:
            record["error"] = content[:1000]  # tools report failures as JSON instead of raising
        logger.info("Tool %s finished in %.0fs", record["tool"], record["duration_ms"] / 1000)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw: Any) -> None:
        self._finish(run_id, output=output)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        self._finish(run_id, error=error)
