from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import error as urllib_error
from urllib import request

from .config import (
    LLMFrontendConfig,
    default_base_url_for_provider,
    default_model_for_provider,
    normalize_provider,
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(Protocol):
    def complete_json(self, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        ...


class LLMChatClient:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_config(
        cls,
        config: LLMFrontendConfig,
        model: str | None = None,
        provider: str | None = None,
    ) -> "LLMChatClient":
        resolved_provider = normalize_provider(
            provider or os.environ.get(config.llm_provider_env, config.provider)
        )
        api_key = os.environ.get(config.llm_api_key_env)
        if not api_key and resolved_provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing required environment variable: {config.llm_api_key_env}")
        resolved_model = (
            model
            or os.environ.get(config.llm_model_env)
            or config.model
            or default_model_for_provider(resolved_provider)
        )
        if resolved_provider == "gemini" and _looks_like_non_gemini_model(resolved_model):
            raise ValueError(
                "Gemini provider selected but model "
                f"{resolved_model!r} does not look like a Gemini model. "
                "Use a Gemini model such as 'gemini-2.5-flash', or omit --model "
                "and let the client pick the default Gemini model."
            )
        base_url = (
            os.environ.get(config.llm_base_url_env)
            or config.llm_base_url
            or default_base_url_for_provider(resolved_provider)
        )
        return cls(
            provider=resolved_provider,
            api_key=api_key,
            model=resolved_model,
            base_url=base_url,
            timeout_s=config.request_timeout_s,
        )

    def complete_json(self, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        if self.provider == "gemini":
            return self._complete_json_gemini(messages, temperature=temperature)
        return self._complete_json_openai_compatible(messages, temperature=temperature)

    def _complete_json_openai_compatible(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [message.to_dict() for message in messages],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        payload = _read_json_response(
            req,
            timeout_s=self.timeout_s,
            provider=self.provider,
            model=self.model,
        )

        message = payload["choices"][0]["message"]["content"]
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

    def _complete_json_gemini(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
    ) -> str:
        system_parts: list[str] = []
        contents: list[dict[str, object]] = []
        for message in messages:
            text = message.content.strip()
            if not text:
                continue
            if message.role == "system":
                system_parts.append(text)
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                {
                    "role": role,
                    "parts": [{"text": text}],
                }
            )

        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        if system_parts:
            payload["system_instruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}],
            }

        if self.model.startswith(("models/", "tunedModels/")):
            model_name = self.model
        else:
            model_name = f"models/{self.model}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/{model_name}:generateContent",
            data=body,
            method="POST",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        response_payload = _read_json_response(
            req,
            timeout_s=self.timeout_s,
            provider=self.provider,
            model=self.model,
        )

        candidates = response_payload.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
            if text_parts:
                return "".join(text_parts).strip()

        prompt_feedback = response_payload.get("promptFeedback")
        if prompt_feedback:
            raise RuntimeError(f"Gemini returned no text candidates: {prompt_feedback}")
        raise RuntimeError("Gemini returned no text candidates.")


def _looks_like_non_gemini_model(model: str) -> bool:
    text = model.strip().lower()
    return text.startswith(("gpt-", "o1", "o3", "o4", "claude", "deepseek"))


def _read_json_response(
    req: request.Request,
    timeout_s: float,
    provider: str,
    model: str,
) -> dict[str, object]:
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body
        try:
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                detail = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        message = (
            f"{provider} request failed with HTTP {exc.code} for model {model!r}: "
            f"{exc.reason}."
        )
        if detail:
            message += f" Response body: {detail}"
        if provider == "gemini" and exc.code == 404:
            message += (
                " This usually means the model name is invalid for Gemini or the "
                "endpoint/model path does not exist. Try a Gemini model such as "
                "'gemini-2.5-flash'."
            )
        raise RuntimeError(message) from exc
