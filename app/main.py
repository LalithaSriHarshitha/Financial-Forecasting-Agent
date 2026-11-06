"""FastAPI application: the public interface of the forecasting agent.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app import __version__
from app.agent.forecast_agent import run_forecast
from app.agent.progress import ProgressLogger
from app.config import get_settings
from app.db.repository import get_request_log, get_tool_calls, log_request, log_tool_calls
from app.db.session import get_engine, init_db
from app.ingestion.manifest import load_manifest
from app.logging_config import setup_logging
from app.schemas.forecast import ForecastRequest, ForecastResponse

logger = logging.getLogger("api")


# Agent runs execute here so a request can be timed out (FORECAST_TIMEOUT_S).
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="forecast")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        init_db()
    except Exception as exc:
        logger.error("Cannot connect to the database. Check the MYSQL_* values in .env (%s)", exc)
        raise
    logger.info("TCS Forecasting Agent v%s ready (model: %s)", __version__, settings.chat_model_name)
    yield


app = FastAPI(
    title="TCS Financial Forecasting Agent",
    version=__version__,
    description="An AI agent that reads TCS financial reports and earnings-call transcripts "
    "and returns a structured qualitative forecast for the next quarter.",
    lifespan=lifespan,
)


def _is_quota_error(exc: BaseException) -> bool:
    text_ = f"{type(exc).__name__} {exc}"
    return "RESOURCE_EXHAUSTED" in text_ or "RateLimit" in text_


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["system"])
def health() -> dict:
    settings = get_settings()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    docs = load_manifest()
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "provider": settings.llm_provider,
        "model": settings.chat_model_name,
        "embedding_model": settings.embedding_model_name,
        "database": "connected" if db_ok else "unreachable",
        "documents_available": sum(d.exists for d in docs),
        "documents_expected": len(docs),
        "vector_index_built": settings.chroma_path.exists(),
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Run the agent on an analytical task and return a structured forecast.

    Every request and its final JSON (or error) is logged to MySQL.
    """
    settings = get_settings()
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    logger.info("Forecast request %s: %s", request_id, req.task[:120])
    progress = ProgressLogger()
    try:
        future = _executor.submit(run_forecast, req.task, progress=progress)
        try:
            run = future.result(timeout=settings.forecast_timeout_s)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(f"forecast exceeded FORECAST_TIMEOUT_S={settings.forecast_timeout_s:.0f}s") from exc
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.exception("Forecast request %s failed", request_id)
        log_tool_calls(request_id, progress.tool_calls)
        log_request(
            request_id=request_id,
            endpoint="/forecast",
            request_payload=req.model_dump(),
            status="error",
            error_message=f"{type(exc).__name__}: {exc}",
            model=settings.chat_model_name,
            latency_ms=latency,
        )
        if isinstance(exc, TimeoutError):
            raise HTTPException(
                status_code=504,
                detail={"request_id": request_id, "error": str(exc), "usage_so_far": progress.usage},
            ) from exc
        if _is_quota_error(exc):
            raise HTTPException(
                status_code=429,
                detail={
                    "request_id": request_id,
                    "error": "LLM API quota exhausted. Wait for the quota to reset, use a model with a "
                    "higher quota, or switch to a local model with LLM_PROVIDER=ollama.",
                    "provider_message": str(exc)[:500],
                },
            ) from exc
        raise HTTPException(
            status_code=500,
            detail={"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    response = ForecastResponse(
        request_id=request_id,
        status="success",
        model=settings.chat_model_name,
        generated_at=datetime.now(timezone.utc),
        latency_ms=int((time.perf_counter() - started) * 1000),
        tool_calls=run.trace,
        grounding=run.grounding,
        usage=run.usage,
        result=run.result,
    )
    log_tool_calls(request_id, run.tool_calls)
    log_request(
        request_id=request_id,
        endpoint="/forecast",
        request_payload=req.model_dump(),
        status="success",
        response_payload=response.model_dump(mode="json"),
        model=settings.chat_model_name,
        latency_ms=response.latency_ms,
    )
    return response


@app.get("/forecast/{request_id}", tags=["forecast"])
def get_logged_forecast(request_id: str) -> dict:
    """Fetch a previously logged request and its response from MySQL."""
    row = get_request_log(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="request_id not found")
    return {
        "request_id": row.request_id,
        "status": row.status,
        "created_at": row.created_at,
        "latency_ms": row.latency_ms,
        "request": row.request_payload,
        "response": row.response_payload,
        "error": row.error_message,
        "tool_calls": [
            {"sequence": c.sequence, "tool": c.tool_name, "input": c.tool_input,
             "duration_ms": c.duration_ms, "output_chars": c.output_chars, "error": c.error}
            for c in get_tool_calls(request_id)
        ],
    }


@app.get("/metrics", tags=["data"])
def financial_metrics(quarter: str | None = None) -> dict:
    """Validated financial metrics extracted from the fact sheets (no LLM call)."""
    from app.tools.financial_extractor import get_financial_metrics

    return get_financial_metrics([quarter] if quarter else None)
