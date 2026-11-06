"""Read/write helpers for request logs. Logging failures never break the API response."""

import logging

from sqlalchemy import select

from app.db.models import RequestLog
from app.db.session import session_scope

logger = logging.getLogger(__name__)


def log_request(
    *,
    request_id: str,
    endpoint: str,
    request_payload: dict,
    status: str,
    response_payload: dict | None = None,
    error_message: str | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
) -> None:
    try:
        with session_scope() as s:
            s.add(
                RequestLog(
                    request_id=request_id,
                    endpoint=endpoint,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    status=status,
                    error_message=error_message,
                    model=model,
                    latency_ms=latency_ms,
                )
            )
    except Exception:  # pragma: no cover - we still want the API to answer
        logger.exception("Failed to write request log %s to the database", request_id)


def get_request_log(request_id: str) -> RequestLog | None:
    with session_scope() as s:
        return s.scalars(select(RequestLog).where(RequestLog.request_id == request_id)).first()
