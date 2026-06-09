import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Note,
  ExportMeetingResponse,
  SessionEvent,
  SessionStatePayload,
  SessionState,
  SummarySnapshot,
  SummaryType,
  TranscriptSegment,
} from "../types";

const DEFAULT_WS = "ws://127.0.0.1:8765/ws/session";
const VOCABULARY_STORAGE_KEY = "meetingbro.vocabulary";
const PREVIEW_FORMAL_COVER_TOLERANCE_SECONDS = 0.3;
const MAX_PREVIEW_STACK = 2;
const STREAM_TICK_MS = 30;
const STREAM_CHARS_PER_TICK = 2;
const STREAM_NEW_CONTENT_WIPE_MS = 80;
const PREVIEW_MIN_DWELL_MS = 550;
const PREVIEW_LEAVE_MS = 700;

function isPreviewCoveredByFormal(preview: TranscriptSegment, formal: TranscriptSegment): boolean {
  return preview.end_time <= formal.end_time + PREVIEW_FORMAL_COVER_TOLERANCE_SECONDS;
}

function mergePreviewStack(stack: TranscriptSegment[], nextPreview: TranscriptSegment): TranscriptSegment[] {
  const trimmedText = nextPreview.text.trim();
  if (!trimmedText) return stack;

  const withoutCoveredDuplicates = stack.filter((item) => {
    const sameWindow =
      Math.abs(item.start_time - nextPreview.start_time) < 0.12 &&
      Math.abs(item.end_time - nextPreview.end_time) < 0.12;
    return !(sameWindow || item.text.trim() === trimmedText);
  });

  const latest = withoutCoveredDuplicates.at(-1);
  if (latest) {
    const startsClose = nextPreview.start_time - latest.start_time < 0.9;
    const endsClose = nextPreview.end_time - latest.end_time < 0.9;
    const latestText = latest.text.trim();
    const shorter = Math.min(latestText.length, trimmedText.length);
    let prefix = 0;
    while (prefix < shorter && latestText[prefix] === trimmedText[prefix]) {
      prefix += 1;
    }
    const similarPrefix = shorter > 0 && prefix / shorter >= 0.72;
    // Also replace when start is close AND text is similar — covers same-sentence
    // extensions where end_time grew beyond the 0.9 s threshold.
    if ((startsClose && endsClose) || (startsClose && similarPrefix) || (similarPrefix && endsClose)) {
      return [...withoutCoveredDuplicates.slice(0, -1), nextPreview].slice(-MAX_PREVIEW_STACK);
    }
  }

  return [...withoutCoveredDuplicates, nextPreview].slice(-MAX_PREVIEW_STACK);
}

// The `window.meetingbro` bridge type lives in ../meetingbro-bridge.d.ts.

export interface SaveNoteInput {
  content: string;
  source_type?: string;
  source_id?: string;
}

export interface ExportMeetingInput {
  source?: string;
  runtime_profile?: string;
  summary_language?: string;
  subtitle_language?: string;
  export_root?: string;
  export_dir?: string;
  bilingual?: boolean;
  target_language?: "zh" | "en" | "de";
}

export interface SessionOptions {
  enabled?: boolean;
  source?: string;           // "mic" | "loopback" | "system" | "mixed"
  summaryLanguage?: string;  // "en" | "zh" | "de"
  speechLanguage?: string;   // "auto" | "en" | "zh" | "de"
  subtitleLanguage?: string; // "off" | "en" | "zh" | "de"
  runtimeProfile?: string;   // "balanced" | "performance" | "summary_only"
  audioDevice?: string;      // sounddevice input index, "" = default
  loopbackDevice?: string;   // system output (speaker) name, "" = default
}

export interface SessionView {
  connected: boolean;
  state: SessionState | "disconnected";
  meetingId: string | null;
  sessionStartedAt: string | null;
  elapsedSeconds: number;
  sessionStats: SessionStatePayload | null;
  segments: TranscriptSegment[];
  previewSegment: TranscriptSegment | null;
  previewSegments: TranscriptSegment[];
  isExperimentalPreview: boolean;
  previewDisplayText: string | null;
  previewIsStreaming: boolean;
  departingPreviewSegments: TranscriptSegment[];
  latestByType: Partial<Record<SummaryType, SummarySnapshot>>;
  historyByType: Partial<Record<SummaryType, SummarySnapshot[]>>;
  notes: Note[];
  lastError: string | null;
  lastErrorCode: string | null;
  saveNote: (input: SaveNoteInput) => Promise<void>;
  saveBookmark: (label?: string) => Promise<void>;
  applyVocabulary: (value: string) => void;
  exportMeeting: (input?: ExportMeetingInput) => Promise<ExportMeetingResponse | null>;
  requestSummary: (summaryType: Extract<SummaryType, "rolling_summary" | "cumulative_meeting_summary" | "refined_transcript">) => void;
  pauseSession: () => void;
  resumeSession: () => void;
  stopSession: () => void;
}

export function useSessionSocket(options: SessionOptions = {}): SessionView {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SessionView["state"]>("disconnected");
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [sessionStartedAt, setSessionStartedAt] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sessionStats, setSessionStats] = useState<SessionStatePayload | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [previewSegment, setPreviewSegment] = useState<TranscriptSegment | null>(null);
  const [previewSegments, setPreviewSegments] = useState<TranscriptSegment[]>([]);
  const [isExperimentalPreview, setIsExperimentalPreview] = useState(false);
  const [latestByType, setLatestByType] = useState<SessionView["latestByType"]>({});
  const [historyByType, setHistoryByType] = useState<SessionView["historyByType"]>({});
  const [notes, setNotes] = useState<Note[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const [lastErrorCode, setLastErrorCode] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const previewTimerRef = useRef<number | null>(null);
  const previewHoldTimerRef = useRef<number | null>(null);
  const queuedPreviewRef = useRef<TranscriptSegment | null>(null);
  const previewSegmentRef = useRef<TranscriptSegment | null>(null);
  const [departingPreviewSegments, setDepartingPreviewSegments] = useState<TranscriptSegment[]>([]);
  const previewSegmentsRef = useRef<TranscriptSegment[]>([]);
  const previewVisibleSinceRef = useRef<Map<string, number>>(new Map());
  const previewDepartScheduledRef = useRef<Set<string>>(new Set());
  const previewDepartTimerRefs = useRef<Set<number>>(new Set());
  const [previewDisplayText, setPreviewDisplayText] = useState<string | null>(null);
  const [previewIsStreaming, setPreviewIsStreaming] = useState(false);
  const streamIntervalRef = useRef<number | null>(null);
  const streamWipeTimerRef = useRef<number | null>(null);
  const streamTargetRef = useRef<string>("");
  const streamPosRef = useRef<number>(0);
  const streamActiveRef = useRef<boolean>(false);

  const enabled = options.enabled ?? true;
  const source = options.source ?? "loopback";
  const summaryLanguage = options.summaryLanguage ?? "en";
  const speechLanguage = options.speechLanguage ?? "auto";
  const subtitleLanguage = options.subtitleLanguage ?? "off";
  const runtimeProfile = options.runtimeProfile ?? "balanced";
  const audioDevice = options.audioDevice ?? "";
  const loopbackDevice = options.loopbackDevice ?? "";
  const effectiveSubtitleLanguage = runtimeProfile === "summary_only" ? "off" : subtitleLanguage;

  useEffect(() => {
    previewSegmentRef.current = previewSegment;
  }, [previewSegment]);

  useEffect(() => {
    previewSegmentsRef.current = previewSegments;
  }, [previewSegments]);

  useEffect(() => {
    const clearQueuedPreview = () => {
      if (previewTimerRef.current != null) {
        window.clearTimeout(previewTimerRef.current);
        previewTimerRef.current = null;
      }
      queuedPreviewRef.current = null;
    };

    const clearPreviewHold = () => {
      if (previewHoldTimerRef.current != null) {
        window.clearTimeout(previewHoldTimerRef.current);
        previewHoldTimerRef.current = null;
      }
    };

    const schedulePreviewTimer = (callback: () => void, delayMs: number) => {
      const timer = window.setTimeout(() => {
        previewDepartTimerRefs.current.delete(timer);
        callback();
      }, delayMs);
      previewDepartTimerRefs.current.add(timer);
      return timer;
    };

    const clearDepartingPreviews = () => {
      previewDepartTimerRefs.current.forEach((timer) => window.clearTimeout(timer));
      previewDepartTimerRefs.current.clear();
      previewDepartScheduledRef.current.clear();
      previewVisibleSinceRef.current.clear();
      setDepartingPreviewSegments([]);
    };

    const rememberPreviewVisible = (segment: TranscriptSegment) => {
      if (!previewVisibleSinceRef.current.has(segment.id)) {
        previewVisibleSinceRef.current.set(segment.id, Date.now());
      }
    };

    const clearStreaming = () => {
      if (streamWipeTimerRef.current != null) {
        window.clearTimeout(streamWipeTimerRef.current);
        streamWipeTimerRef.current = null;
      }
      if (streamIntervalRef.current != null) {
        window.clearInterval(streamIntervalRef.current);
        streamIntervalRef.current = null;
      }
      streamActiveRef.current = false;
      streamTargetRef.current = "";
      streamPosRef.current = 0;
      setPreviewDisplayText(null);
      setPreviewIsStreaming(false);
    };

    const startStreaming = (nextText: string) => {
      if (!nextText) { clearStreaming(); return; }
      if (streamWipeTimerRef.current != null) {
        window.clearTimeout(streamWipeTimerRef.current);
        streamWipeTimerRef.current = null;
      }
      const currentPos = streamPosRef.current;
      const currentDisplay = streamTargetRef.current.slice(0, currentPos);
      const maxLen = Math.min(currentDisplay.length, nextText.length);
      let prefixLen = 0;
      while (prefixLen < maxLen && currentDisplay[prefixLen] === nextText[prefixLen]) {
        prefixLen += 1;
      }
      // When the new text shares little with what we already displayed, it's a new
      // sentence - wipe briefly, then stream it from the start.
      const similarity = maxLen > 0 ? prefixLen / maxLen : (currentDisplay.length === 0 ? 1.0 : 0.0);
      const isNewContent = similarity < 0.3 && currentDisplay.length > 0;
      const newPos = isNewContent ? 0 : Math.max(currentPos, prefixLen);

      const beginStreaming = (startPos: number) => {
        streamTargetRef.current = nextText;
        if (startPos >= nextText.length) {
          if (streamIntervalRef.current != null) {
            window.clearInterval(streamIntervalRef.current);
            streamIntervalRef.current = null;
          }
          streamActiveRef.current = false;
          streamPosRef.current = nextText.length;
          setPreviewDisplayText(nextText);
          setPreviewIsStreaming(false);
          return;
        }

        streamPosRef.current = startPos;
        setPreviewDisplayText(nextText.slice(0, startPos) || null);
        if (streamActiveRef.current) {
          // Interval already running - refs are updated, it picks up automatically.
          if (isNewContent) setPreviewIsStreaming(true);
          return;
        }

        streamActiveRef.current = true;
        setPreviewIsStreaming(true);
        streamIntervalRef.current = window.setInterval(() => {
          const target = streamTargetRef.current;
          const pos = streamPosRef.current;
          if (pos >= target.length) {
            window.clearInterval(streamIntervalRef.current!);
            streamIntervalRef.current = null;
            streamActiveRef.current = false;
            setPreviewIsStreaming(false);
            return;
          }
          const nextPos = Math.min(pos + STREAM_CHARS_PER_TICK, target.length);
          streamPosRef.current = nextPos;
          setPreviewDisplayText(target.slice(0, nextPos));
        }, STREAM_TICK_MS);
      };

      if (isNewContent) {
        if (streamIntervalRef.current != null) {
          window.clearInterval(streamIntervalRef.current);
          streamIntervalRef.current = null;
        }
        streamActiveRef.current = false;
        streamTargetRef.current = nextText;
        streamPosRef.current = 0;
        setPreviewIsStreaming(true);
        setPreviewDisplayText(null);
        streamWipeTimerRef.current = window.setTimeout(() => {
          streamWipeTimerRef.current = null;
          beginStreaming(0);
        }, STREAM_NEW_CONTENT_WIPE_MS);
        return;
      }

      beginStreaming(newPos);
    };

    const holdCommittedPreview = (segment: TranscriptSegment) => {
      clearPreviewHold();
      clearStreaming();
      clearDepartingPreviews();
      streamTargetRef.current = segment.text;
      streamPosRef.current = segment.text.length;
      setPreviewDisplayText(segment.text);
      rememberPreviewVisible(segment);
      setPreviewSegment(segment);
      setPreviewSegments([segment]);
      previewHoldTimerRef.current = window.setTimeout(() => {
        setPreviewSegment((prev) => {
          if (!prev) return prev;
          if (prev.id !== segment.id) return prev;
          return null;
        });
        setPreviewSegments((prev) => prev.filter((item) => item.id !== segment.id));
        clearStreaming();
        previewHoldTimerRef.current = null;
      }, 500);
    };

    // Move segments covered by a formal commit into the departing queue so they
    // can play a CSS exit animation before being removed from the DOM. Very new
    // preview rows get a short minimum dwell time to avoid blink-in/blink-out noise.
    const queueDepartingPreviews = (segmentsToDepart: TranscriptSegment[]) => {
      if (segmentsToDepart.length === 0) {
        return;
      }
      const ids = segmentsToDepart.map((segment) => segment.id);
      setPreviewSegments((prev) => prev.filter((item) => !ids.includes(item.id)));
      setDepartingPreviewSegments((prev) => {
        const existingIds = new Set(prev.map((item) => item.id));
        const next = segmentsToDepart.filter((item) => !existingIds.has(item.id));
        return next.length > 0 ? [...prev, ...next] : prev;
      });
      schedulePreviewTimer(() => {
        setDepartingPreviewSegments((prev) => prev.filter((item) => !ids.includes(item.id)));
        ids.forEach((id) => {
          previewDepartScheduledRef.current.delete(id);
          previewVisibleSinceRef.current.delete(id);
        });
      }, PREVIEW_LEAVE_MS);
    };

    const departPreviewSegments = (coveredByFormal: TranscriptSegment) => {
      const now = Date.now();
      const readyToDepart: TranscriptSegment[] = [];
      const covered = previewSegmentsRef.current.filter(
        (item) => isPreviewCoveredByFormal(item, coveredByFormal) && !previewDepartScheduledRef.current.has(item.id),
      );

      for (const segment of covered) {
        previewDepartScheduledRef.current.add(segment.id);
        const visibleSince = previewVisibleSinceRef.current.get(segment.id) ?? now;
        const remainingDwellMs = Math.max(0, PREVIEW_MIN_DWELL_MS - (now - visibleSince));
        if (remainingDwellMs <= 0) {
          readyToDepart.push(segment);
        } else {
          schedulePreviewTimer(() => queueDepartingPreviews([segment]), remainingDwellMs);
        }
      }

      queueDepartingPreviews(readyToDepart);
    };

    const schedulePreviewUpdate = (nextPreview: TranscriptSegment | null) => {
      if (nextPreview == null) {
        clearQueuedPreview();
        clearStreaming();
        clearDepartingPreviews();
        setPreviewSegment(null);
        setPreviewSegments([]);
        setIsExperimentalPreview(false);
        return;
      }

      const currentPreview = previewSegmentRef.current;
      const currentText = currentPreview?.text.trim() ?? "";
      const nextText = nextPreview.text.trim();
      const currentWindow = currentPreview
        ? `${currentPreview.start_time}:${currentPreview.end_time}`
        : "";
      const nextWindow = `${nextPreview.start_time}:${nextPreview.end_time}`;

      if (currentText === nextText && currentWindow === nextWindow) {
        return;
      }

      if (queuedPreviewRef.current?.id === nextPreview.id && queuedPreviewRef.current.text === nextPreview.text) {
        return;
      }

      clearQueuedPreview();

      const sharedPrefixLength = (() => {
        const max = Math.min(currentText.length, nextText.length);
        let index = 0;
        while (index < max && currentText[index] === nextText[index]) {
          index += 1;
        }
        return index;
      })();
      const sharesMostContent =
        currentText.length > 0 &&
        sharedPrefixLength >= Math.floor(Math.min(currentText.length, nextText.length) * 0.7);
      const updateDelayMs = sharesMostContent ? 140 : 220;

      queuedPreviewRef.current = nextPreview;
      previewTimerRef.current = window.setTimeout(() => {
        rememberPreviewVisible(nextPreview);
        setPreviewSegment(nextPreview);
        setPreviewSegments((prev) => mergePreviewStack(prev, nextPreview));
        startStreaming(nextPreview.text);
        previewTimerRef.current = null;
        queuedPreviewRef.current = null;
      }, updateDelayMs);
    };

    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      clearQueuedPreview();
      clearPreviewHold();
      clearStreaming();
      clearDepartingPreviews();
      setPreviewSegment(null);
      setPreviewSegments([]);
      setIsExperimentalPreview(false);
      setConnected(false);
      setState("disconnected");
      return;
    }

    setLastError(null);
    setLastErrorCode(null);
    setSegments([]);
    setElapsedSeconds(0);
    setSessionStats(null);
    clearQueuedPreview();
    clearPreviewHold();
    clearStreaming();
    clearDepartingPreviews();
    setPreviewSegment(null);
    setPreviewSegments([]);
    setIsExperimentalPreview(false);
    setLatestByType({});
    setHistoryByType({});
    setNotes([]);
    setMeetingId(null);
    setSessionStartedAt(null);
    setState("starting");

    const base = window.meetingbro?.backendWs ?? "ws://127.0.0.1:8765";
    const params = new URLSearchParams({ source, summary_language: summaryLanguage, runtime_profile: runtimeProfile });
    if (speechLanguage !== "auto") params.set("forced_language", speechLanguage);
    if (audioDevice) params.set("audio_device", audioDevice);
    if (loopbackDevice) params.set("loopback_device", loopbackDevice);
    const vocabulary = window.localStorage.getItem(VOCABULARY_STORAGE_KEY)?.trim();
    if (vocabulary) params.set("vocabulary_hint", vocabulary);
    const url = `${base}/ws/session?${params.toString()}`;
    const ws = new WebSocket(url.startsWith("ws") ? url : DEFAULT_WS);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setState("disconnected");
    };
    ws.onerror = () => setLastError("websocket error - check that the backend is running and the selected audio source is available");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as SessionEvent;
        switch (msg.type) {
          case "session_state":
            setState(msg.payload.state);
            setMeetingId(msg.payload.meeting_id);
            setElapsedSeconds(msg.payload.elapsed_seconds);
            setSessionStats(msg.payload);
            if (msg.payload.state === "running") {
              setSessionStartedAt((prev) => prev ?? new Date().toISOString());
            }
            break;
          case "transcript_segment":
            clearPreviewHold();
            if (queuedPreviewRef.current && isPreviewCoveredByFormal(queuedPreviewRef.current, msg.payload)) {
              clearQueuedPreview();
            }
            if (previewSegmentRef.current && isPreviewCoveredByFormal(previewSegmentRef.current, msg.payload)) {
              clearStreaming();
            }
            setPreviewSegment((prev) => {
              if (!prev) return prev;
              const matchesCommitted =
                prev.text.trim() === msg.payload.text.trim() &&
                Math.abs(prev.start_time - msg.payload.start_time) < 0.01 &&
                Math.abs(prev.end_time - msg.payload.end_time) < 0.01;
              if (matchesCommitted) {
                setIsExperimentalPreview(false);
                holdCommittedPreview(msg.payload);
                return msg.payload;
              }
              if (isPreviewCoveredByFormal(prev, msg.payload)) {
                setIsExperimentalPreview(false);
                return null;
              }
              return prev;
            });
            departPreviewSegments(msg.payload);
            setSegments((prev) => [...prev, msg.payload]);
            break;
          case "transcript_segment_removed":
            setSegments((prev) => prev.filter((s) => s.id !== msg.payload.segment_id));
            break;
          case "transcript_translation":
            setSegments((prev) => prev.map((segment) => {
              if (segment.id !== msg.payload.segment_id) {
                return segment;
              }
              return {
                ...segment,
                translations: {
                  ...segment.translations,
                  [msg.payload.language]: msg.payload.text,
                },
              };
            }));
            break;
          case "transcript_preview":
            setIsExperimentalPreview(msg.payload.preview_is_experimental ?? false);
            schedulePreviewUpdate(msg.payload.segment);
            break;
          case "summary_snapshot":
            setLatestByType((prev) => ({
              ...prev,
              [msg.payload.summary_type]: msg.payload,
            }));
            setHistoryByType((prev) => {
              const existing = prev[msg.payload.summary_type] ?? [];
              if (existing.some((snap) => snap.id === msg.payload.id)) {
                return prev;
              }
              return {
                ...prev,
                [msg.payload.summary_type]: [...existing, msg.payload],
              };
            });
            break;
          case "note_saved":
            setNotes((prev) => [...prev, msg.payload]);
            break;
          case "error":
            setLastError(msg.payload.message);
            setLastErrorCode(msg.payload.code);
            break;
        }
      } catch (e) {
        setLastError(`parse error: ${(e as Error).message}`);
      }
    };

    return () => {
      clearQueuedPreview();
      clearPreviewHold();
      clearStreaming();
      clearDepartingPreviews();
      ws.close();
    };
  }, [enabled]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!enabled || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(
      JSON.stringify({
        type: "update_settings",
        payload: {
          source,
          summary_language: summaryLanguage,
          forced_language: speechLanguage,
          subtitle_language: effectiveSubtitleLanguage,
          runtime_profile: runtimeProfile,
          audio_device: audioDevice,
          loopback_device: loopbackDevice,
        },
      }),
    );
  }, [connected, effectiveSubtitleLanguage, enabled, source, speechLanguage, summaryLanguage, runtimeProfile, audioDevice, loopbackDevice]);

  const stopSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }, []);

  const pauseSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "pause" }));
    }
  }, []);

  const resumeSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "resume" }));
    }
  }, []);

  const saveNote = useCallback(
    async (input: SaveNoteInput) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !meetingId) {
        setLastError("cannot save note — session not ready");
        return;
      }
      ws.send(
        JSON.stringify({
          type: "save_note",
          payload: {
            content: input.content,
            source_type: input.source_type,
            source_id: input.source_id,
          },
        }),
      );
    },
    [meetingId],
  );

  const applyVocabulary = useCallback((value: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(
      JSON.stringify({
        type: "update_settings",
        payload: {
          vocabulary_hint: value.trim(),
        },
      }),
    );
  }, []);

  const requestSummary = useCallback((summaryType: Extract<SummaryType, "rolling_summary" | "cumulative_meeting_summary" | "refined_transcript">) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setLastError("cannot refresh summary — session not ready");
      return;
    }
    ws.send(
      JSON.stringify({
        type: "request_summary",
        payload: { summary_type: summaryType },
      }),
    );
  }, []);

  const saveBookmark = useCallback(
    async (label = "") => saveNote({ content: label, source_type: "bookmark" }),
    [saveNote],
  );

  const exportMeeting = useCallback(
    async (input: ExportMeetingInput = {}) => {
      if (!meetingId) {
        setLastError("cannot export meeting — session not ready");
        return null;
      }
      const base = window.meetingbro?.backendHttp ?? "http://127.0.0.1:8765";
      const params = new URLSearchParams();
      if (input.bilingual) {
        params.set("bilingual", "true");
      }
      if (input.target_language) {
        params.set("target_language", input.target_language);
      }
      const url = `${base}/meetings/${meetingId}/export${params.size ? `?${params.toString()}` : ""}`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      if (!response.ok) {
        const message = `export failed (${response.status})`;
        setLastError(message);
        return null;
      }
      return await response.json() as ExportMeetingResponse;
    },
    [meetingId],
  );

  return {
    connected,
    state,
    meetingId,
    sessionStartedAt,
    elapsedSeconds,
    sessionStats,
    segments,
    previewSegment,
    previewSegments,
    isExperimentalPreview,
    previewDisplayText,
    previewIsStreaming,
    departingPreviewSegments,
    latestByType,
    historyByType,
    notes,
    lastError,
    lastErrorCode,
    saveNote,
    saveBookmark,
    applyVocabulary,
    exportMeeting,
    requestSummary,
    pauseSession,
    resumeSession,
    stopSession,
  };
}
