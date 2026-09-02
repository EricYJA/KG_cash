from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

LLMVendor = Literal["openai", "google", "tamu"]

DEFAULT_LLM_VENDOR: LLMVendor = "tamu"
LLM_VENDOR_CHOICES: tuple[LLMVendor, ...] = ("openai", "google", "tamu")
LLM_API_KEY_ENV = "LLM_API_KEY"
# A whole pool of keys, comma/newline separated. This is how the pool reaches a
# container: the .env_keys file lives on the host and is not mounted, so the
# runner reads it and forwards the values in this one variable.
LLM_API_KEYS_ENV = "LLM_API_KEYS"
# Override the location of the key file; otherwise it is searched for by name.
LLM_KEYS_FILE_ENV = "LLM_KEYS_FILE"
KEYS_FILENAME = ".env_keys"

# Which names in .env_keys belong to which vendor. A file that holds only one
# vendor's keys does not have to say so -- see _keys_from_file.
VENDOR_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "tamu": ("TAMUS", "TAMU"),
    "openai": ("OPENAI", "GPT"),
    "google": ("GOOGLE", "GEMINI"),
}


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
    # Every key the client may fall back to, best candidate first. `api_key` is
    # this tuple's first entry; it stays a plain field so existing callers that
    # only ever wanted one key keep working.
    api_keys: tuple[str, ...] = ()


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
        default_model="protected.Claude-Haiku-4.5",
        default_base_url="https://chat-api.tamu.ai/openai",
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
    """Resolve runtime LLM settings from CLI overrides, the key pool, and presets."""

    preset = get_llm_preset(vendor or DEFAULT_LLM_VENDOR)
    resolved_api_keys = resolve_api_keys(preset.vendor, api_key)
    if not resolved_api_keys:
        raise RuntimeError(
            "Missing API key for vendor "
            f"{preset.vendor!r}. Pass --API_KEY/--api-key, set {LLM_API_KEY_ENV}"
            f" or {LLM_API_KEYS_ENV}, or put keys in a {KEYS_FILENAME} file."
        )
    resolved_api_key = resolved_api_keys[0]
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
        api_keys=resolved_api_keys,
    )


def resolve_api_keys(vendor: str, api_key: str | None = None) -> tuple[str, ...]:
    """Ordered pool of API keys to try for `vendor`, best candidate first.

    An explicit key (CLI `--api-key`) wins outright: asking for one key means
    one key. Otherwise the pool is assembled, in order, from LLM_API_KEYS, the
    .env_keys file, and the single LLM_API_KEY, deduped with order preserved.
    The client walks this pool when a key answers 400/5xx, so the order is the
    fallback order.
    """
    explicit = _first_non_empty(api_key)
    if explicit:
        return (explicit,)

    keys: list[str] = []
    keys += _split_keys(os.environ.get(LLM_API_KEYS_ENV))
    keys += _keys_from_file(vendor)
    single = _first_non_empty(os.environ.get(LLM_API_KEY_ENV))
    if single:
        keys.append(single)
    return tuple(dict.fromkeys(keys))


def _split_keys(raw: str | None) -> list[str]:
    """Split a comma/whitespace separated key list into its non-empty entries."""
    if not raw:
        return []
    parts = (chunk.strip() for chunk in raw.replace(",", "\n").split())
    return [part for part in parts if part]


def keys_file_path() -> Path | None:
    """Locate the .env_keys file: LLM_KEYS_FILE if set, else the nearest one.

    "Nearest" is searched upwards from the working directory and then from this
    module, so it is found whether the caller runs from the repo root or from a
    project subdirectory (ToG and RoG both chdir into their own tree).
    """
    override = _first_non_empty(os.environ.get(LLM_KEYS_FILE_ENV))
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in roots:
        for directory in (root, *root.parents):
            candidate = directory / KEYS_FILENAME
            if candidate.is_file():
                return candidate
    return None


def _keys_from_file(vendor: str) -> list[str]:
    """Read this vendor's keys out of .env_keys, in NAME order (KEY2 before KEY10)."""
    path = keys_file_path()
    if path is None:
        return []

    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            entries.append((name.strip(), value))

    prefixes = VENDOR_KEY_PREFIXES.get(vendor, ())
    matched = [
        entry
        for entry in entries
        if any(entry[0].upper().startswith(prefix) for prefix in prefixes)
    ]
    # No name matched: treat the file as a pool for whatever vendor is running,
    # so a plain KEY1=... / KEY2=... file works without vendor-specific naming.
    selected = matched or entries
    selected.sort(key=lambda entry: _key_sort_key(entry[0]))
    return [value for _, value in selected]


def _key_sort_key(name: str) -> tuple[str, int, str]:
    """Sort NAME1..NAME10 numerically rather than lexicographically."""
    stem = name.rstrip("0123456789")
    digits = name[len(stem):]
    return (stem.upper(), int(digits) if digits else -1, name)


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
