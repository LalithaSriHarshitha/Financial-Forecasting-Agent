"""Factory functions for the Gemini chat model and embeddings (one place to swap providers)."""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings


def get_chat_model(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=s.google_api_key,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_retries=3,
        timeout=120,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=s.gemini_embedding_model, google_api_key=s.google_api_key
    )
