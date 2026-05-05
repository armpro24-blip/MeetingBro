from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Any

_DOTENV_LOADED = False

_LOGGER = logging.getLogger(__name__)


def _load_dotenv_if_present() -> None:
    """Load a local .env file without adding a runtime dependency.

    Supports both common dotenv lines:
        MEETINGBRO_LLM_API_KEY=...

    and PowerShell-style lines users often paste into .env on Windows:
        $env:MEETINGBRO_LLM_API_KEY="..."

    Existing process environment variables win over .env values.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[4] / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            item = line.strip()
            if not item or item.startswith("#") or "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            if key.lower().startswith("$env:"):
                key = key[5:]
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 45.0


class OpenAICompatibleClient:
    """Tiny stdlib client for OpenAI-compatible chat-completions APIs.

    This avoids requiring the OpenAI SDK just to use providers such as LongCat.
    It is intentionally non-streaming because MeetingBro's summary/translation
    workers need one complete result per snapshot.
    """

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._cfg = config

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        """Build a client from MeetingBro/LongCat environment variables.

        LongCat shortcut:
          LONGCAT_API_KEY=<key>

        Generic OpenAI-compatible config:
          MEETINGBRO_LLM_API_KEY=<key>
          MEETINGBRO_LLM_BASE_URL=https://api.longcat.chat/openai
          MEETINGBRO_LLM_MODEL=LongCat-Flash-Chat
        """

        _load_dotenv_if_present()
        api_key = os.environ.get("MEETINGBRO_LLM_API_KEY") or os.environ.get(
            "LONGCAT_API_KEY"
        )
        if not api_key:
            return None

        base_url = os.environ.get("MEETINGBRO_LLM_BASE_URL")
        if not base_url:
            base_url = (
                "https://api.longcat.chat/openai"
                if os.environ.get("LONGCAT_API_KEY")
                else "https://api.openai.com"
            )

        model = os.environ.get("MEETINGBRO_LLM_MODEL")
        if not model:
            model = (
                "LongCat-Flash-Chat"
                if "longcat.chat" in base_url
                else os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            )

        timeout = float(os.environ.get("MEETINGBRO_LLM_TIMEOUT_SECONDS", "45"))
        return cls(
            OpenAICompatibleConfig(
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                model=model,
                timeout_seconds=timeout,
            )
        )

    @property
    def model(self) -> str:
        return self._cfg.model

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        base_url = self._cfg.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.time()
        _LOGGER.debug("LLM request start model=%s url=%s", self._cfg.model, url)
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            _LOGGER.warning(
                "LLM HTTP error model=%s url=%s code=%s detail=%s",
                self._cfg.model,
                url,
                exc.code,
                detail,
            )
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        finally:
            duration = time.time() - start
            _LOGGER.debug("LLM request finished model=%s duration=%.3fs", self._cfg.model, duration)

        data = json.loads(raw)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = ""
        # Primary: standard OpenAI-style chat message content
        if isinstance(message, dict):
            content = message.get("content") or message.get("reasoning") or message.get("text") or ""
        # Fallbacks for other provider shapes
        if not content:
            content = choice.get("text") if isinstance(choice, dict) else ""
        content = (content or "").strip()
        if not content:
            _LOGGER.warning(
                "LLM returned empty content for model=%s; raw_response=%s",
                self._cfg.model,
                raw if len(raw) < 4000 else raw[:4000] + "...",
            )
        return content
