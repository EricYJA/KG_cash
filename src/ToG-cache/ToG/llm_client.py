from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from typing import Protocol

import httpx

from llm_config import ResolvedLLMConfig


# Statuses that mean "this key is no good, try the next one" rather than "this
# request is malformed". Vendors are inconsistent about which one they use for a
# revoked/over-quota key -- TAMU answers 400 -- so the whole 4xx band that could
# plausibly be key-related rotates, as does every 5xx.
KEY_ROTATION_STATUS_CODES = frozenset({400, 401, 402, 403, 408, 429})


# A 200 whose body carries no usable message happens: a content filter, an
# upstream hiccup, a truncated proxy response. It is transient far more often
# than not, so it is retried (on a fresh key) rather than raised at the caller.
# This is a floor, not a cap: complete_json() lifts it to the pool size so every
# key gets a turn (see there).
MAX_RESPONSE_ATTEMPTS = 3
RESPONSE_RETRY_BACKOFF_S = 1.0
# The backoff grows with the attempt, so cap it: on a ten-key pool the linear
# form would sit out most of a minute before spending the last key.
MAX_RESPONSE_RETRY_BACKOFF_S = 5.0


class LLMRequestError(RuntimeError):
    """A request that produced no usable answer, with what is known kept for routing.

    `status_code` is 0 when no answer arrived at all; `body` is then empty.
    """

    def __init__(self, message: str, status_code: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    @property
    def short(self) -> str:
        """One-phrase form for the "which keys failed" tally."""
        return f"HTTP {self.status_code}" if self.status_code else "no usable answer"


class LLMHTTPError(LLMRequestError):
    """A non-2xx answer from the LLM endpoint, with the status kept for routing."""

    def __init__(self, status_code: int, message: str, body: str) -> None:
        super().__init__(message, status_code=status_code, body=body)


class LLMTransportError(LLMRequestError):
    """A request that never came back as a usable answer: timeout, dropped
    connection, unparseable body.

    Rotated over exactly like a bad status. The cause may well not be the key --
    but the next key is the only lever this client has, and a long run trying it
    beats a run that stops with nine keys unspent.
    """

    @property
    def short(self) -> str:
        # A status may be carried for the dump (a 200 holding junk), but what
        # failed here is the answer, not the status -- say so.
        return "no usable answer"


class LLMKeyPoolExhaustedError(RuntimeError):
    """Every key in the pool was tried for one request and every one failed.

    Not the vendor having a bad minute on one call: the client walked the whole
    pool back to back and got nothing, so there is no key left to answer the
    next question either. A run that carries on from here only fills its output
    with unanswered questions.

    Narrow on purpose. Everything else -- a malformed response, a status that
    rotation could not fix -- stays a plain RuntimeError and is absorbed one
    question at a time, exactly as before.

    Subclasses RuntimeError so every existing `except RuntimeError` still
    catches it; callers that want to stop check for this type first.
    """


class LLMMalformedResponseError(RuntimeError):
    """A 2xx answer whose body carried no usable message content.

    Subclasses RuntimeError so callers that already catch RuntimeError -- which
    is what the experiment scripts do to keep one bad question from killing a
    run -- keep working unchanged.
    """


def _should_rotate_key(status_code: int) -> bool:
    return status_code in KEY_ROTATION_STATUS_CODES or status_code >= 500


# Which key each (endpoint, pool) last succeeded on. ToG builds a fresh client
# per LLM call, so without a module-level cursor every call would walk over the
# same dead keys again; RoG keeps one client and would be fine either way.
_KEY_CURSOR: dict[tuple[str, tuple[str, ...]], int] = {}


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
        # Every key we may fall back to, plus the one currently in use. Keeping
        # `api_key` pointing at the live key means callers reading it (logs,
        # dumps) see the key that actually served the last request.
        self.api_keys = tuple(connection_config.api_keys) or (
            connection_config.api_key,
        )
        self.model = connection_config.model
        self.base_url = _normalize_base_url(connection_config.base_url)
        self.api_base_url = self.base_url
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.timeout_s = timeout_s
        self._cursor_key = (self.chat_completions_url, self.api_keys)
        self._key_index = _KEY_CURSOR.get(self._cursor_key, 0)
        self.api_key = self.api_keys[self._key_index]

    @classmethod
    def from_connection_config(
        cls,
        connection_config: ResolvedLLMConfig,
        timeout_s: float,
    ) -> "LLMChatClient":
        return cls(connection_config=connection_config, timeout_s=timeout_s)

    def _post_with_key_rotation(self, payload: dict[str, object]) -> dict[str, object]:
        """POST `payload`, moving to the next API key on any failed request, and retry.

        The walk starts at the key that last worked, so a dead key is stepped
        over once rather than re-tried on every call, and it wraps around so no
        key in the pool is skipped. Only once every key has failed does this
        raise -- which is the point: a long experiment run should burn through
        the pool before it gives up, not stop on the first exhausted key. A
        timeout or a dropped connection rotates as well as a 400/5xx: the fault
        is often the key's rate limiter refusing to talk, and even when it is
        not, spending the pool costs one run nothing.
        """
        failures: list[str] = []
        last_error: LLMRequestError | None = None

        for offset in range(len(self.api_keys)):
            index = (self._key_index + offset) % len(self.api_keys)
            try:
                response = _post_json(
                    url=self.chat_completions_url,
                    payload=payload,
                    vendor=self.vendor,
                    api_key=self.api_keys[index],
                    timeout_s=self.timeout_s,
                    model=self.model,
                )
            except LLMRequestError as exc:
                if isinstance(exc, LLMHTTPError) and not _should_rotate_key(
                    exc.status_code
                ):
                    # e.g. 404 unknown model: every key would answer the same.
                    raise RuntimeError(
                        self._with_dump_hint(str(exc), payload, exc)
                    ) from exc
                failures.append(f"key {index + 1}: {exc.short}")
                last_error = exc
                remaining = len(self.api_keys) - offset - 1
                print(
                    f"[llm_client] {self.vendor} key {index + 1}/{len(self.api_keys)}"
                    f" failed with {exc.short};"
                    + (
                        f" trying the next key ({remaining} left)"
                        if remaining
                        else " no keys left"
                    ),
                    flush=True,
                )
                continue

            # Stick with whichever key answered, for the next call and for the
            # next client built off the same pool.
            self._key_index = index
            self.api_key = self.api_keys[index]
            _KEY_CURSOR[self._cursor_key] = index
            return response

        message = (
            f"{self.vendor} request failed for model {self.model!r} on all"
            f" {len(self.api_keys)} API keys ({'; '.join(failures)})."
        )
        if last_error is not None:
            message += f" Last error: {last_error}"
        raise LLMKeyPoolExhaustedError(
            self._with_dump_hint(message, payload, last_error)
        ) from last_error

    def _with_dump_hint(
        self,
        message: str,
        payload: dict[str, object],
        error: LLMRequestError | None,
    ) -> str:
        """Append the replayable request dump to `message`, best effort.

        A dump failure must never mask the real error, so any OSError here is
        reported and swallowed.
        """
        try:
            dump_dir = _dump_failed_request(
                vendor=self.vendor,
                url=self.chat_completions_url,
                payload=payload,
                status_code=error.status_code if error is not None else 0,
                body=error.body if error is not None else "",
            )
        except OSError as exc:
            print(f"[llm_client] warning: could not write failed-request dump: {exc}")
            return message
        return (
            f"{message} Request dump:"
            f" {dump_dir / f'{self.vendor}_last_request.json'}"
            f" Replay script: {dump_dir / f'replay_{self.vendor}_last_request.sh'}"
        )

    def complete_json(
        self, messages: list[ChatMessage], temperature: float = 0.0
    ) -> str:
        """Send one chat completion and return its text.

        A body that arrives without usable content is retried on a fresh key,
        because that failure is usually a transient hiccup on the vendor's side
        and re-asking is enough. The retry budget is the whole key pool (floored
        at MAX_RESPONSE_ATTEMPTS, so a single-key setup still gets its retries):
        an exhausted key that answers 200-with-nothing must not end the call with
        the other nine keys unspent. Only once the budget is gone does the error
        reach the caller, which is where a run decides to record the question as
        unanswered rather than abort.
        """
        payload = {
            "model": self.model,
            "messages": _prepare_messages(messages),
            "stream": False,
        }
        if temperature != 0.0:
            payload["temperature"] = temperature

        max_attempts = max(MAX_RESPONSE_ATTEMPTS, len(self.api_keys))
        for attempt in range(1, max_attempts + 1):
            response = self._post_with_key_rotation(payload)
            try:
                message = _extract_message_content(response)
            except LLMMalformedResponseError as exc:
                if attempt == max_attempts:
                    raise
                print(
                    f"[llm_client] {self.vendor} attempt {attempt}"
                    f"/{max_attempts}: {exc} Retrying on the next key.",
                    flush=True,
                )
                self._advance_key()
                time.sleep(
                    min(
                        RESPONSE_RETRY_BACKOFF_S * attempt,
                        MAX_RESPONSE_RETRY_BACKOFF_S,
                    )
                )
                continue
            return _flatten_content(message)

        raise AssertionError("unreachable: the loop above either returns or raises")

    def _advance_key(self) -> None:
        """Move to the next key in the pool, so a retry does not repeat this one."""
        if len(self.api_keys) < 2:
            return
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self._key_index]
        _KEY_CURSOR[self._cursor_key] = self._key_index


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
        raise LLMMalformedResponseError(
            "LLM response did not include any choices."
        )

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMMalformedResponseError(
            "LLM response choice payload was not an object."
        )

    message = first_choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise LLMMalformedResponseError(
            "LLM response message payload did not include content."
        )

    return message["content"]


def _flatten_content(message: object) -> str:
    """Collapse a content field -- string, or a list of content parts -- to text."""
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
        # Rotatable, not fatal: the caller tries this on the next key. A key
        # whose rate limiter stops answering looks exactly like this.
        raise LLMTransportError(
            f"{vendor} request timed out after {timeout_s} seconds for model {model!r}."
        ) from exc
    except httpx.HTTPError as exc:
        detail = str(exc).strip()
        message = f"{vendor} request failed for model {model!r}."
        if detail:
            message += f" Detail: {detail}"
        raise LLMTransportError(message) from exc

    status_code = response.status_code
    body = response.text.strip()
    if status_code >= 400:
        # Raised, not dumped: the caller decides whether this status is worth
        # another key, and only dumps once the whole pool is spent.
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
        raise LLMHTTPError(status_code=status_code, message=message, body=body)

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        # A 200 carrying something that is not JSON (a proxy error page, a
        # truncated stream) says nothing about the request -- try the next key.
        raise LLMTransportError(
            f"{vendor} returned invalid JSON for model {model!r}: {body}",
            status_code=status_code,
            body=body,
        ) from exc


def _dump_base_dir() -> Path:
    """Pick a writable directory for failed-request dumps.

    Honours LLM_DUMP_DIR; else ./artifacts when writable; else a temp dir. The
    container often runs with cwd on a read-only mount (e.g. /rog), so a relative
    ./artifacts is not writable -- fall back rather than crash on the dump and
    mask the real error.
    """
    override = os.environ.get("LLM_DUMP_DIR")
    if override:
        return Path(override)
    candidate = Path("artifacts")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        if os.access(candidate, os.W_OK):
            return candidate
    except OSError:
        pass
    return Path(tempfile.gettempdir()) / "llm_failed_requests"


def _dump_failed_request(
    vendor: str,
    url: str,
    payload: dict[str, object],
    status_code: int,
    body: str,
) -> Path:
    dump_dir = _dump_base_dir()
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
