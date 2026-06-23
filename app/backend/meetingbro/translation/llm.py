from __future__ import annotations

import logging
import os
from typing import Optional

from ..llm import OpenAICompatibleClient
from ..schemas import LanguageCode
from .base import Translator
from .passthrough import PassthroughTranslator

logger = logging.getLogger(__name__)

_LANG_NAMES = {"en": "English", "zh": "Chinese", "de": "German"}

_SYSTEM_PROMPT = (
    "You are a precise translator. Translate the following text from {src} to {tgt}. "
    "Output ONLY the translated text, nothing else. Preserve the original meaning, "
    "tone, and formatting. Do not add explanations or notes."
)


class LLMTranslator(Translator):
    """LLM-backed translator supporting Chinese, English, and German.

    Uses an OpenAI-compatible provider such as LongCat when ``LONGCAT_API_KEY``
    or ``MEETINGBRO_LLM_API_KEY`` is present, or Anthropic/OpenAI SDKs when
    their provider-specific keys are present. Falls back to
    ``PassthroughTranslator`` otherwise, so the pipeline never breaks and the
    original transcript remains the source of truth regardless.
    """

    def __init__(self, *, fallback: Optional[Translator] = None) -> None:
        self._fallback = fallback or PassthroughTranslator()
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self._openai_key = os.environ.get("OPENAI_API_KEY")
        self._anthropic_client = None
        self._openai_client = None
        self._compatible_client = OpenAICompatibleClient.from_env()
        # Last provider error, mirroring LLMSummarizer.last_error so the session
        # manager can surface a silent translation fallback to the UI. None means
        # the most recent attempt succeeded (or no LLM call was attempted).
        self.last_error: Optional[str] = None

    def _ensure_client(self) -> Optional[str]:
        if self._compatible_client is not None:
            return "openai_compatible"
        if self._anthropic_key and self._anthropic_client is None:
            try:
                import anthropic  # type: ignore

                self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
                return "anthropic"
            except Exception as exc:
                logger.warning("anthropic client unavailable: %s", exc)
        if self._openai_key and self._openai_client is None:
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=self._openai_key)
                return "openai"
            except Exception as exc:
                logger.warning("openai client unavailable: %s", exc)
        if self._anthropic_client is not None:
            return "anthropic"
        if self._openai_client is not None:
            return "openai"
        return None

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: LanguageCode,
    ) -> str:
        if not text or source_language == target_language:
            return text

        provider = self._ensure_client()
        if provider is None:
            return self._fallback.translate(
                text, source_language=source_language, target_language=target_language
            )

        src_name = _LANG_NAMES.get(source_language, source_language)
        tgt_name = _LANG_NAMES.get(target_language, target_language)
        system_prompt = _SYSTEM_PROMPT.format(src=src_name, tgt=tgt_name)

        try:
            if provider == "openai_compatible":
                assert self._compatible_client is not None
                result = self._compatible_client.chat(
                    system=system_prompt,
                    user=text,
                    max_tokens=1024,
                    temperature=0.0,
                )
            elif provider == "anthropic":
                assert self._anthropic_client is not None
                msg = self._anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}],
                )
                result = "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text"
                ).strip()
            else:
                assert self._openai_client is not None
                resp = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=1024,
                )
                result = (resp.choices[0].message.content or "").strip()
            self.last_error = None
            return result
        except Exception as exc:
            logger.warning("LLM translation failed, falling back: %s", exc)
            self.last_error = str(exc)
            return self._fallback.translate(
                text, source_language=source_language, target_language=target_language
            )
