"""Chat-model and embedding factories: the single place where the LLM provider is chosen.

LLM_PROVIDER=ollama -> local models via Ollama (no API key, no quotas)
LLM_PROVIDER=gemini -> Google Gemini API
"""

import warnings
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.config import get_settings

# Newer Gemini models ignore temperature; the library warns on every call.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter:
    """One limiter shared by every Gemini call (agent + both tools), so a forecast
    stays under the requests-per-minute quota instead of failing with 429."""
    rpm = max(1, get_settings().llm_requests_per_minute)
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=2)


def get_chat_model(temperature: float | None = None, for_tools: bool = False) -> BaseChatModel:
    """Chat model for the agent, or (for_tools=True) for the tools' internal calls."""
    s = get_settings()
    temp = s.llm_temperature if temperature is None else temperature
    model = s.tool_model_name if for_tools else s.chat_model_name

    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=s.ollama_base_url,
            temperature=temp,
            num_ctx=s.ollama_num_ctx,
            reasoning=False,  # skip "thinking" output: faster, and keeps JSON clean
            keep_alive="10m",
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=s.google_api_key,
        temperature=temp,
        rate_limiter=_rate_limiter(),
        max_retries=6,
        timeout=120,
    )


def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.ollama_embedding_model, base_url=s.ollama_base_url)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(model=s.gemini_embedding_model, google_api_key=s.google_api_key)
