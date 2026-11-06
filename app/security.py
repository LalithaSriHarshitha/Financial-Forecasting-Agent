"""Access control for a publicly deployed service.

* API key: when API_KEYS is set, POST /forecast and GET /forecast/{id} require an
  `X-API-Key` header matching one of the keys. With API_KEYS empty (local use) the API is open.
* Rate limit: at most FORECAST_RATE_LIMIT_PER_HOUR forecasts per client per rolling hour,
  so a shared link cannot exhaust the LLM quota. Clients are identified by API key, else IP.
"""

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Required when the server sets API_KEYS.")


def require_api_key(key: str | None = Security(api_key_header)) -> str | None:
    keys = get_settings().api_key_set
    if not keys:
        return None
    if key and any(hmac.compare_digest(key, k) for k in keys):
        return key
    raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")  # set by Render's proxy
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, limit: int, window_s: float = 3600) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > window_s:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(window_s - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit: {limit} forecasts per hour. Try again in {retry} s.",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


forecast_limiter = RateLimiter()


def limit_forecasts(request: Request, key: str | None = Depends(require_api_key)) -> None:
    client = f"key:{key[:8]}" if key else f"ip:{_client_ip(request)}"
    forecast_limiter.check(client, get_settings().forecast_rate_limit_per_hour)
