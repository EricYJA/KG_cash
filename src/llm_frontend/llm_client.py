from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Protocol

import httpx

from .llm_config import ResolvedLLMConfig


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(Protocol):
    def complete_json(
        self, messages: list[ChatMessage], temperature: float = 0.0
    ) -> str: ...


class LLMChatClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(
        self,
        connection_config: ResolvedLLMConfig,
        timeout_s: float,
    ) -> None:
        self.vendor = connection_config.vendor
        self.api_name = connection_config.vendor
        self.api_key = connection_config.api_key
        self.model = connection_config.model
        self.base_url = _normalize_base_url(connection_config.base_url)
        self.api_base_url = self.base_url
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.timeout_s = timeout_s

    @classmethod
    def from_connection_config(
        cls,
        connection_config: ResolvedLLMConfig,
        timeout_s: float,
    ) -> "LLMChatClient":
        return cls(connection_config=connection_config, timeout_s=timeout_s)

    def complete_json(
        self, messages: list[ChatMessage], temperature: float = 0.0
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": _prepare_messages(messages),
        }
        if temperature != 0.0:
            payload["temperature"] = temperature
        payload = _post_json(
            url=self.chat_completions_url,
            payload=payload,
            vendor=self.vendor,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            model=self.model,
        )

        message = _extract_message_content(payload)
        if isinstance(message, str):
            return message.strip()
        if isinstance(message, list):
            parts = []
            for item in message:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(message).strip()


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("LLM base URL must not be empty.")
    return base_url


def _prepare_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []

    for message in messages:
        content = _compact_text(message.content)
        if not content:
            continue

        role = (
            message.role if message.role in {"system", "user", "assistant"} else "user"
        )
        prepared.append({"role": role, "content": content})

    return prepared or [{"role": "user", "content": ""}]


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _extract_message_content(payload: dict[str, object]) -> object:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include any choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("LLM response choice payload was not an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise RuntimeError("LLM response message payload did not include content.")

    return message["content"]


def _post_json(
    url: str,
    payload: dict[str, object],
    vendor: str,
    api_key: str,
    timeout_s: float,
    model: str,
) -> dict[str, object]:
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"{vendor} request timed out after {timeout_s} seconds for model {model!r}."
        ) from exc
    except httpx.HTTPError as exc:
        detail = str(exc).strip()
        message = f"{vendor} request failed for model {model!r}."
        if detail:
            message += f" Detail: {detail}"
        raise RuntimeError(message)

    status_code = response.status_code
    body = response.text.strip()
    if status_code >= 400:
        dump_dir = _dump_failed_request(
            vendor=vendor,
            url=url,
            payload=payload,
            status_code=status_code,
            body=body,
        )
        detail = body
        try:
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                detail = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        message = (
            f"{vendor} request failed with HTTP {status_code} for model {model!r}."
        )
        if detail:
            message += f" Response body: {detail}"
        message += (
            " Request dump:"
            f" {dump_dir / f'{vendor}_last_request.json'}"
            f" Replay script: {dump_dir / f'replay_{vendor}_last_request.sh'}"
        )
        raise RuntimeError(message)

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{vendor} returned invalid JSON for model {model!r}: {body}"
        ) from exc


def _dump_failed_request(
    vendor: str,
    url: str,
    payload: dict[str, object],
    status_code: int,
    body: str,
) -> Path:
    dump_dir = Path("artifacts")
    dump_dir.mkdir(parents=True, exist_ok=True)

    payload_path = (dump_dir / f"{vendor}_last_request.json").resolve()
    meta_path = (dump_dir / f"{vendor}_last_request_meta.json").resolve()
    script_path = (dump_dir / f"replay_{vendor}_last_request.sh").resolve()

    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "vendor": vendor,
                "url": url,
                "status_code": status_code,
                "response_body": body,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                ': "${API_KEY:?API_KEY is required}"',
                f"curl -sS -X POST {shlex.quote(url)} \\",
                '  -H "Authorization: Bearer ${API_KEY}" \\',
                '  -H "Content-Type: application/json" \\',
                f"  --data-binary @{shlex.quote(str(payload_path))} | jq",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return dump_dir.resolve()
