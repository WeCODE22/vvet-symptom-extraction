"""OpenAI-compatible chat client. Tests pass a fake instead of this."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text for one farmer message."""


class LLMError(Exception):
    """The provider could not return text. Callers should not treat this as 'no signs'."""


class OpenAIChatLLM:
    """Talks to any host that implements POST /chat/completions."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        json_mode: bool = True,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.json_mode = json_mode
        self.timeout = timeout
        self._client = client

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set")

        payload: dict = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            try:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPError as exc:
                raise LLMError(f"LLM request failed: {exc}") from exc
            except ValueError as exc:
                raise LLMError("LLM response was not JSON") from exc
        finally:
            if owns_client:
                client.close()

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response did not include message text") from exc

        text = _message_text(content)
        if not text:
            raise LLMError("LLM response did not include message text")
        return text


def llm_from_env() -> OpenAIChatLLM:
    _load_project_env()
    json_mode = _env_flag("LLM_JSON_MODE", default=True)
    return OpenAIChatLLM(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        json_mode=json_mode,
    )


def _load_project_env() -> None:
    """Load this project's .env. Existing environment variables win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts).strip()
    return ""
