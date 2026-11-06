"""Factory functions for the Gemini chat model and embeddings (one place to swap providers)."""

import warnings
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings

# Newer Gemini models ignore temperature; the library warns on every call.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter:
    """One limiter shared by every chat call (agent + both tools), so a forecast
    stays under the Gemini requests-per-minute quota instead of failing with 429."""
    rpm = max(1, get_settings().llm_requests_per_minute)
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=2)


def get_chat_model(temperature: float | None = None, for_tools: bool = False) -> ChatGoogleGenerativeAI:
    """Chat model for the agent, or (for_tools=True) for the tools' internal calls."""
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=(s.gemini_tool_model or s.gemini_model) if for_tools else s.gemini_model,
        google_api_key=s.google_api_key,
        temperature=s.llm_temperature if temperature is None else temperature,
        rate_limiter=_rate_limiter(),
        max_retries=6,
        timeout=120,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=s.gemini_embedding_model, google_api_key=s.google_api_key
    )
