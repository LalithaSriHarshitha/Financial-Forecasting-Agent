"""FastAPI application: the public interface of the forecasting agent.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from app import __version__
from app.agent.forecast_agent import run_forecast
from app.config import get_settings
from app.db.repository import get_request_log, log_request
from app.db.session import get_engine, init_db
from app.ingestion.manifest import load_manifest
from app.logging_config import setup_logging
from app.schemas.forecast import ForecastRequest, ForecastResponse

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        init_db()
    except Exception as exc:
        logger.error("Cannot connect to the database. Check the MYSQL_* values in .env (%s)", exc)
        raise
    logger.info("TCS Forecasting Agent v%s ready (model: %s)", __version__, settings.gemini_model)
    yield


app = FastAPI(
    title="TCS Financial Forecasting Agent",
    version=__version__,
    description="An AI agent that reads TCS financial reports and earnings-call transcripts "
    "and returns a structured qualitative forecast for the next quarter.",
    lifespan=lifespan,
)


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
        "model": settings.gemini_model,
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
    try:
        result, trace = run_forecast(req.task)
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.exception("Forecast request %s failed", request_id)
        log_request(
            request_id=request_id,
            endpoint="/forecast",
            request_payload=req.model_dump(),
            status="error",
            error_message=f"{type(exc).__name__}: {exc}",
            model=settings.gemini_model,
            latency_ms=latency,
        )
        raise HTTPException(
            status_code=500,
            detail={"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    response = ForecastResponse(
        request_id=request_id,
        status="success",
        model=settings.gemini_model,
        generated_at=datetime.now(timezone.utc),
        latency_ms=int((time.perf_counter() - started) * 1000),
        tool_calls=trace,
        result=result,
    )
    log_request(
        request_id=request_id,
        endpoint="/forecast",
        request_payload=req.model_dump(),
        status="success",
        response_payload=response.model_dump(mode="json"),
        model=settings.gemini_model,
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
    }
