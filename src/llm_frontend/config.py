from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROVIDER = "gemini"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def normalize_provider(value: str | None) -> str:
    text = str(value or DEFAULT_PROVIDER).strip().lower()
    aliases = {
        "openai": "openai_compatible",
        "openai-compatible": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "gemini": "gemini",
        "google": "gemini",
        "google-gemini": "gemini",
        "google_gemini": "gemini",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported LLM provider: {value}")
    return aliases[text]


def default_model_for_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_OPENAI_COMPATIBLE_MODEL


def default_base_url_for_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized == "gemini":
        return DEFAULT_GEMINI_BASE_URL
    return DEFAULT_OPENAI_COMPATIBLE_BASE_URL


@dataclass(frozen=True)
class LLMFrontendConfig:
    provider: str = DEFAULT_PROVIDER
    model: str | None = None
    temperature: float = 0.0
    max_steps: int = 6
    max_memory_steps: int = 6
    max_frontier_entities: int = 8
    max_relation_candidates: int = 12
    relation_scan_limit: int = 12
    repeat_query_limit: int = 2
    repeat_frontier_limit: int = 3
    fallback_answer_limit: int = 10
    request_timeout_s: float = 60.0
    llm_provider_env: str = "LLM_PROVIDER"
    llm_api_key_env: str = "LLM_API_KEY"
    llm_model_env: str = "LLM_MODEL"
    llm_base_url_env: str = "LLM_BASE_URL"
    llm_base_url: str | None = None
