from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

LLMVendor = Literal["openai", "google", "tamu"]

DEFAULT_LLM_VENDOR: LLMVendor = "google"
LLM_VENDOR_CHOICES: tuple[LLMVendor, ...] = ("openai", "google", "tamu")
LLM_API_KEY_ENV = "LLM_API_KEY"


@dataclass(frozen=True)
class LLMPresetConfig:
    """Preset connection defaults for one supported LLM vendor."""

    vendor: LLMVendor
    default_model: str
    default_base_url: str


@dataclass(frozen=True)
class ResolvedLLMConfig:
    """Fully resolved connection settings used by the runtime client."""

    vendor: LLMVendor
    api_key: str
    model: str
    base_url: str


LLM_PRESET_CONFIGS: dict[str, LLMPresetConfig] = {
    "openai": LLMPresetConfig(
        vendor="openai",
        default_model="gpt-4.1-mini",
        default_base_url="https://api.openai.com/v1",
    ),
    "google": LLMPresetConfig(
        vendor="google",
        # default_model="gemini-2.5-flash",
        default_model="gemini-3-flash-preview",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    "tamu": LLMPresetConfig(
        vendor="tamu",
        default_model="protected.gemini-2.0-flash-lite",
        default_base_url="https://chat-api.tamu.ai/api",
    ),
}


def get_llm_preset(vendor: str) -> LLMPresetConfig:
    """Return the preset config for a supported vendor name."""

    normalized_vendor = vendor.strip().lower()
    try:
        return LLM_PRESET_CONFIGS[normalized_vendor]
    except KeyError as exc:
        supported = ", ".join(LLM_VENDOR_CHOICES)
        raise ValueError(
            f"Unsupported vendor {vendor!r}. Choose one of: {supported}."
        ) from exc


def resolve_llm_config(
    vendor: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ResolvedLLMConfig:
    """Resolve runtime LLM settings from CLI overrides, one API-key env var, and presets."""

    preset = get_llm_preset(vendor or DEFAULT_LLM_VENDOR)
    resolved_api_key = _first_non_empty(api_key, os.environ.get(LLM_API_KEY_ENV))
    if not resolved_api_key:
        raise RuntimeError(
            "Missing API key for vendor "
            f"{preset.vendor!r}. Pass --API_KEY/--api-key or set {LLM_API_KEY_ENV}."
        )
    resolved_model = _first_non_empty(model, preset.default_model)
    if resolved_model is None:
        raise RuntimeError(
            f"Resolved model for vendor {preset.vendor!r} must not be empty."
        )
    resolved_base_url = _first_non_empty(base_url, preset.default_base_url)
    if resolved_base_url is None:
        raise RuntimeError(
            f"Resolved base URL for vendor {preset.vendor!r} must not be empty."
        )
    return ResolvedLLMConfig(
        vendor=preset.vendor,
        api_key=resolved_api_key,
        model=resolved_model,
        base_url=resolved_base_url,
    )


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
