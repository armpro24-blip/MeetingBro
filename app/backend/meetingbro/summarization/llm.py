from __future__ import annotations

import json
import logging
import os
import re
from typing import Sequence

from ..llm import OpenAICompatibleClient
from ..schemas import LanguageCode, TranscriptSegment
from .base import Summarizer, SummaryKind
from .heuristic import HeuristicSummarizer

logger = logging.getLogger(__name__)


_PROMPTS = {
    "meeting_memory": (
        "You maintain a compressed, structured memory of an ongoing meeting in {language}. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and do not ignore non-English text. "
        "Update the previous memory using only the new transcript. Keep it concise and "
        "bounded. Preserve durable facts; remove resolved or redundant details. Output "
        "Markdown with these exact headings: ## Topics, ## Decisions, ## Action Items, "
        "## Open Questions, ## Important Facts. Do not invent content."
    ),
    "rolling_summary": (
        "You are producing an incremental live update for a meeting. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "You will receive two inputs: (1) the previous rolling summary, which is already done and must NOT "
        "be repeated or paraphrased, and (2) new transcript since that summary. "
        "Your job is ONLY to report what is genuinely new in input (2): new facts, updated numbers, "
        "new decisions, newly assigned actions, reversals, or additions not present in (1). "
        "Write 2-4 sentences in {language} covering only this new content. "
        "If the new transcript contains nothing that wasn't already covered in the previous summary, "
        "say so in one sentence. Never restate or paraphrase the previous summary. Do not invent content."
    ),
    "cumulative_meeting_summary": (
        "You are maintaining a live meeting board for an ongoing meeting. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "Use the compressed meeting memory as the durable context and the recent "
        "transcript as fresh evidence. Output Markdown with these exact headings: "
        "## Meeting State, ## Decisions, ## Action Items, ## Open Questions. "
        "Under each heading, use short bullet points in {language}. Keep it compact, "
        "stable, and cumulative from the start of the meeting to now. If a section has "
        "no evidence yet, write '- None yet'. Do not invent content."
    ),
    "final_summary": (
        "You are writing the final recap of a meeting. Use the compressed meeting "
        "memory as the durable context and the recent transcript as fresh evidence. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "In {language}, produce: key points, decisions, action items, open questions. "
        "Be faithful to the provided context; do not invent content."
    ),
    "refined_transcript": (
        "You are producing a clean conversation record for an ongoing meeting in {language}. "
        "You may receive evidence from both Whisper formal transcript segments and "
        "Qwen preview hints. Do NOT assume either engine is always better: decide "
        "case by case from timestamps, overlap, language consistency, surrounding "
        "context, and plausibility. If Qwen is more plausible for a local phrase, "
        "use it; if Whisper is more plausible, use Whisper; if they conflict and "
        "neither is clearly reliable, keep the safer wording and mark the issue "
        "as uncertain. Never invent facts, speakers, decisions, names, or timestamps. "
        "Preserve the meeting's mixed Chinese, English, and German content. This is "
        "not a summary: keep it as a readable dialogue/transcript from the beginning "
        "of the supplied conversation to the end. Output Markdown with exactly these "
        "headings: ## Clean Conversation and ## Uncertain / Conflicts. Under "
        "Clean Conversation, use chronological timestamped dialogue-style bullets. "
        "Under Uncertain / Conflicts, list only uncertain gaps/conflicts or write "
        "'- None noticed'."
    ),
}

_FINALIZE_PROMPT = (
    "You are finalizing a meeting in {language}. Given the compressed meeting memory, a recent transcript tail, and known vocabulary, "
    "output ONLY valid JSON with the exact shape: "
    '{{"chapters":[{{"title":"<<8 words","time_start":0,"time_end":0,"summary":"<<3 sentences"}}],'
    '"action_items":[{{"text":"...","assignee":"<name or null>","due":"<date or null>"}}],'
    '"final_summary":"5-10 sentences in {language} covering decisions, key points, open questions"}}. '
    "Do NOT invent content. Use the vocabulary list as exact spellings of names/terms. "
    "The transcript may mix Chinese, English, German. Earlier meeting content has been condensed into the compressed meeting memory above. "
    "The recent transcript tail provides verbatim grounding for the most recent minutes; older topics, decisions, and action items should be derived from the compressed memory. "
    "For chapters that fall outside the verbatim tail's time range, provide best-effort approximate time_start and time_end based on the meeting's overall structure; precise sub-second accuracy is not required."
)

_LANG_NAMES = {"en": "English", "zh": "Chinese", "de": "German"}


class LLMSummarizer(Summarizer):
    """LLM-backed summarizer.

    Activated when a supported API key is present in the environment. For
    LongCat, set ``LONGCAT_API_KEY``; the default endpoint/model are:
    ``https://api.longcat.chat/openai`` and ``LongCat-Flash-Chat``.

    Falls back to the heuristic summarizer otherwise, so the pipeline always
    produces transcript-driven output.
    """

    def __init__(self, *, fallback: Summarizer | None = None) -> None:
        self._fallback = fallback or HeuristicSummarizer()
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self._openai_key = os.environ.get("OPENAI_API_KEY")
        self._anthropic_client = None
        self._openai_client = None
        self._compatible_client = OpenAICompatibleClient.from_env()
        # Set to the most recent provider error when an LLM call fell back to the
        # heuristic summarizer; None after a successful call. The SessionManager
        # reads this to surface a visible (deduped) warning to the UI.
        self.last_error: str | None = None

    def _ensure_client(self):
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

    def summarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        kind: SummaryKind,
        language: LanguageCode,
        previous_summary: str | None = None,
        vocabulary: str | None = None,
    ) -> str:
        if not segments:
            return ""
        provider = self._ensure_client()
        if provider is None:
            return self._fallback.summarize(
                segments,
                kind=kind,
                language=language,
                previous_summary=previous_summary,
                vocabulary=vocabulary,
            )

        transcript_text = "\n".join(
            f"[{s.start_time:.1f}-{s.end_time:.1f}] {s.text}" for s in segments
        )
        system_prompt = _PROMPTS[kind].format(language=_LANG_NAMES.get(language, language))
        user_content = transcript_text
        if previous_summary:
            if kind == "meeting_memory":
                previous_label = "Previous meeting memory"
                new_label = "New transcript to fold into memory"
            elif kind == "rolling_summary":
                previous_label = "Previous rolling summary (already covered — do NOT repeat)"
                new_label = "New transcript since last summary (focus on this)"
            elif kind == "refined_transcript":
                previous_label = "Cross-engine evidence and conflict notes"
                new_label = "Raw ASR transcript to turn into a clean conversation record"
            else:
                previous_label = "Compressed meeting memory / prior summary"
                new_label = "Recent transcript tail"
            user_content = f"{previous_label}:\n{previous_summary}\n\n{new_label}:\n{transcript_text}"
        if vocabulary and vocabulary.strip():
            user_content = (
                "Known names / terms (use these exact spellings if they appear): "
                f"{vocabulary.strip()}\n\n{user_content}"
            )

        try:
            self.last_error = None
            if provider == "openai_compatible":
                assert self._compatible_client is not None
                return self._compatible_client.chat(
                    system=system_prompt,
                    user=user_content,
                    max_tokens=1600 if kind == "refined_transcript" else (900 if kind == "meeting_memory" else 600),
                    temperature=0.2,
                )
            elif provider == "anthropic":
                assert self._anthropic_client is not None
                msg = self._anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=900 if kind == "meeting_memory" else 600,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
                return "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text"
                ).strip()
            else:
                assert self._openai_client is not None
                resp = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM summarization failed, falling back: %s", exc)
            self.last_error = str(exc)
            return self._fallback.summarize(
                segments,
                kind=kind,
                language=language,
                previous_summary=previous_summary,
                vocabulary=vocabulary,
            )

    def finalize_meeting(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        language: LanguageCode,
        vocabulary: str | None = None,
        meeting_memory: str | None = None,
    ) -> dict:
        if not segments:
            return {"chapters": [], "action_items": [], "final_summary": ""}
        provider = self._ensure_client()
        if provider is None:
            return self._fallback.finalize_meeting(
                segments,
                language=language,
                vocabulary=vocabulary,
                meeting_memory=meeting_memory,
            )
        transcript_text = "\n".join(
            f"[{int(s.start_time // 60):02d}:{int(s.start_time % 60):02d}-{int(s.end_time // 60):02d}:{int(s.end_time % 60):02d}] {(s.speaker_id or 'Speaker')}: {s.text}"
            for s in segments if s.text.strip()
        )
        if len(transcript_text) > 30000:
            transcript_text = transcript_text[:13000] + "\n[...truncated for length...]\n" + transcript_text[-13000:]
        user_content = (
            f"Known vocabulary: {vocabulary.strip() if vocabulary and vocabulary.strip() else '(none provided)'}\n\n"
            "Compressed meeting memory (durable context):\n"
            f"{meeting_memory.strip() if meeting_memory and meeting_memory.strip() else '(empty)'}\n\n"
            f"Recent transcript tail (verbatim, with timestamps):\n{transcript_text}"
        )
        system_prompt = _FINALIZE_PROMPT.format(language=_LANG_NAMES.get(language, language))
        if provider == "openai_compatible":
            assert self._compatible_client is not None
            raw = self._compatible_client.chat(system=system_prompt, user=user_content, max_tokens=1400, temperature=0.1)
        elif provider == "anthropic":
            assert self._anthropic_client is not None
            msg = self._anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        else:
            assert self._openai_client is not None
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = match.group(0) if match else raw
        result = json.loads(payload)
        if not isinstance(result, dict) or not isinstance(result.get("chapters"), list) or not isinstance(result.get("action_items"), list):
            raise ValueError("finalize_meeting returned invalid JSON shape")
        final_summary = result.get("final_summary")
        if not isinstance(final_summary, str) or not final_summary.strip():
            raise ValueError("finalize_meeting returned empty final_summary")
        return {
            "chapters": result["chapters"],
            "action_items": result["action_items"],
            "final_summary": final_summary.strip(),
        }
