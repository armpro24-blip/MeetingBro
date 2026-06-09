import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { ExportMeetingResponse, SummarySnapshot, TranscriptSegment } from "./types";
import { useSessionSocket } from "./session/useSessionSocket";
import type { ExportMeetingInput } from "./session/useSessionSocket";
import { useBackendStatus } from "./system/useBackendStatus";
import { BackendGate } from "./components/BackendGate";
import { SettingsModal } from "./components/SettingsModal";
import { fetchAudioDevices } from "./system/backendApi";
import type { AudioDevices } from "./system/backendApi";

const VOCABULARY_STORAGE_KEY = "meetingbro.vocabulary";
const AUDIO_DEVICE_KEY = "meetingbro.audioDevice";
const LOOPBACK_DEVICE_KEY = "meetingbro.loopbackDevice";
const PREVIEW_HANDOFF_TOLERANCE_SECONDS = 0.35;
const FORMAL_SEGMENT_STAGGER_MS = 75;
const FORMAL_CELL_STAGGER_MS = 55;
const PREVIEW_ROW_STAGGER_MS = 45;
const PREVIEW_CELL_STAGGER_MS = 45;

type StaggerStyle = CSSProperties & {
  "--segment-delay"?: string;
  "--segment-text-delay"?: string;
  "--preview-status-delay"?: string;
  "--preview-text-delay"?: string;
};

function hasPreviewHandoff(segment: TranscriptSegment, previews: readonly TranscriptSegment[]): boolean {
  return previews.some((preview) => (
    preview.end_time <= segment.end_time + PREVIEW_HANDOFF_TOLERANCE_SECONDS &&
    preview.end_time >= segment.start_time - PREVIEW_HANDOFF_TOLERANCE_SECONDS
  ));
}


function formatRange(start: number, end: number): string {
  const fmt = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    const sec = Math.floor(s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };
  return `${fmt(start)} – ${fmt(end)}`;
}

function formatApproxClock(iso: string | null, seconds: number): string {
  if (!iso) return formatRange(seconds, seconds).slice(0, 5);
  const dt = new Date(new Date(iso).getTime() + seconds * 1000);
  return dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatApproxClockRange(iso: string | null, start: number, end: number): string {
  return `${formatApproxClock(iso, start)} – ${formatApproxClock(iso, end)}`;
}

function formatCreatedAt(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatLagSeconds(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  return `${Math.round(seconds)}s`;
}

function getDelayTone(seconds: number | null): "ok" | "warn" | "danger" | "unknown" {
  if (seconds == null) return "unknown";
  if (seconds < 1.2) return "ok";
  if (seconds < 2.8) return "warn";
  return "danger";
}

type AdviceSeverity = "ok" | "warn" | "danger";

interface SystemAdvice {
  severity: AdviceSeverity;
  title: string;
  detail: string;
  action?: string;
}

function getConfidenceTone(confidence: number): "stable" | "soft" | "uncertain" {
  if (confidence < 0.48) return "uncertain";
  if (confidence < 0.67) return "soft";
  return "stable";
}

function getConfidenceLabel(confidence: number): string | null {
  const tone = getConfidenceTone(confidence);
  if (tone === "uncertain") return "uncertain";
  if (tone === "soft") return "low conf";
  return null;
}

function isSystemHealthOnlyMessage(message: string | null): boolean {
  if (!message) return false;
  const normalized = message.toLowerCase();
  return (
    normalized.includes("asr is close to falling behind") ||
    normalized.includes("temporarily skipping retry") ||
    normalized.includes("asr has been processing") ||
    normalized.includes("audio input is buffered") ||
    normalized.includes("backlog")
  );
}

function formatElapsedSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60).toString().padStart(2, "0");
  const s = Math.floor(total % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function formatGain(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(2)}x`;
}

function formatRtf(value: number | null | undefined): string {
  if (value == null) return "???";
  return `${value.toFixed(2)}x`;
}

function getGainTone(
  effectiveMicrophoneGain: number | null | undefined,
  baseMicrophoneGain: number | null | undefined,
  activeSource: string | null | undefined,
): "ok" | "warn" | "danger" | "unknown" {
  if (activeSource !== "mixed") return "unknown";
  if (effectiveMicrophoneGain == null || baseMicrophoneGain == null || baseMicrophoneGain <= 0) {
    return "unknown";
  }
  const boostRatio = effectiveMicrophoneGain / baseMicrophoneGain;
  if (boostRatio < 1.15) return "ok";
  if (boostRatio < 1.45) return "warn";
  return "danger";
}

function formatSourceLabel(source: string | null | undefined): string {
  switch (source) {
    case "mic":
      return "Microphone";
    case "loopback":
    case "system":
      return "System Audio";
    case "mixed":
      return "System + Mic";
    default:
      return source ?? "—";
  }
}

function formatLanguageLabel(language: "zh" | "en" | "de" | null | undefined): string {
  switch (language) {
    case "zh":
      return "Chinese";
    case "en":
      return "English";
    case "de":
      return "German";
    default:
      return "Off";
  }
}

function formatRuntimeProfileLabel(profile: string | null | undefined): string {
  switch (profile) {
    case "summary_only":
      return "Summary Only";
    case "performance":
    case "robust":
      return "Performance";
    case "balanced":
      return "Balanced";
    default:
      return profile ?? "Balanced";
  }
}

function defaultExportFolderName(): string {
  const now = new Date();
  const pad = (value: number, width = 2) => value.toString().padStart(width, "0");
  const timestamp = [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
  ].join("-") + "_" + [
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join("-") + `-${pad(now.getMilliseconds(), 3)}`;
  return `${timestamp}_meetingbro`;
}

function parseSnapshotList(content: string): Array<Record<string, unknown>> {
  try {
    const value = JSON.parse(content);
    return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => !!item && typeof item === "object") : [];
  } catch {
    return [];
  }
}

function getMixedGainHint(
  tone: "ok" | "warn" | "danger" | "unknown",
  activeSource: string | null | undefined,
): string {
  if (activeSource !== "mixed") {
    return "mixed diagnostics appear when source is System + Mic";
  }
  if (tone === "danger") {
    return "mic seems weak · consider raising MEETINGBRO_MIXED_MIC_GAIN";
  }
  if (tone === "warn") {
    return "moderate mic compensation active · monitor voice clarity";
  }
  if (tone === "ok") {
    return "mic and system levels look balanced";
  }
  return "waiting for mixed-level diagnostics";
}

function isNearBottom(element: HTMLDivElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 120;
}

interface SummaryPanelProps {
  title: string;
  subtitle: string;
  snapshot: SummarySnapshot | undefined;
  history: SummarySnapshot[];
  sessionStartedAt: string | null;
  accent: string;
  className?: string;
  compact?: boolean;
  onSaveToNotes: (snapshot: SummarySnapshot) => Promise<void>;
  onRefresh?: () => void;
  refreshDisabled?: boolean;
  refreshBusy?: boolean;
}

function SummaryPanel({ title, subtitle, snapshot, history, sessionStartedAt, accent, className, compact = false, onSaveToNotes, onRefresh, refreshDisabled = false, refreshBusy = false }: SummaryPanelProps) {
  const displayedHistory = history.length > 0 ? history : snapshot ? [snapshot] : [];
  const latestSnapshot = displayedHistory.at(-1) ?? snapshot;
  const range = latestSnapshot
    ? formatApproxClockRange(sessionStartedAt, latestSnapshot.time_start, latestSnapshot.time_end)
    : "—";
  const relativeRange = latestSnapshot
    ? formatRange(latestSnapshot.time_start, latestSnapshot.time_end)
    : "—";
  const createdAt = latestSnapshot
    ? new Date(latestSnapshot.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  const bodyRef = useRef<HTMLDivElement>(null);
  const autoFollowRef = useRef(true);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    if (autoFollowRef.current && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [displayedHistory.length, latestSnapshot?.id]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const onCopy = async () => {
    if (!latestSnapshot) return;
    await navigator.clipboard.writeText(latestSnapshot.content);
  };
  const onExpand = () => {
    if (!latestSnapshot) return;
    setExpanded(true);
  };
  const onSave = async () => {
    if (!latestSnapshot) return;
    await onSaveToNotes(latestSnapshot);
  };

  const bodyJSX = (
    <div
      ref={bodyRef}
      className="panel-body summary-history"
      onScroll={(e) => {
        autoFollowRef.current = isNearBottom(e.currentTarget);
      }}
    >
      {displayedHistory.length === 0 ? (
        "waiting for the first snapshot…"
      ) : (
        displayedHistory.map((item, idx) => {
          const itemRange = formatApproxClockRange(sessionStartedAt, item.time_start, item.time_end);
          const itemRelativeRange = formatRange(item.time_start, item.time_end);
          const itemCreatedAt = new Date(item.created_at).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          });
          const isLatest = idx === displayedHistory.length - 1;
          return (
            <article key={item.id} className={`summary-history-item ${isLatest ? "latest" : ""}`.trim()}>
              <div className="summary-history-meta">
                <span>{itemRange}</span>
                <span>{itemRelativeRange}</span>
                <span>{itemCreatedAt}</span>
              </div>
              <div>{item.content}</div>
            </article>
          );
        })
      )}
    </div>
  );

  const footerJSX = (
    <footer>
      <button onClick={onRefresh} disabled={!onRefresh || refreshDisabled} className={refreshBusy ? "button-busy" : ""} aria-busy={refreshBusy}>
        {refreshBusy ? "Refreshing…" : "Refresh"}
      </button>
      <button onClick={onCopy} disabled={!latestSnapshot}>Copy Latest</button>
      <button onClick={onExpand} disabled={!latestSnapshot}>Expand</button>
      <button onClick={onSave} disabled={!latestSnapshot}>Save Latest</button>
    </footer>
  );

  const modalJSX = expanded ? (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-label={`${title} history`} onClick={() => setExpanded(false)}>
      <div className="modal modal--summary" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2>{title} <span className="modal__range">{range}</span></h2>
          <button type="button" className="modal__close" onClick={() => setExpanded(false)} aria-label="Close">×</button>
        </div>
        <div className="modal__body summary-history">
          {displayedHistory.length === 0 ? (
            "No snapshots yet."
          ) : (
            displayedHistory.map((item, idx) => (
              <article key={item.id} className="summary-history-item">
                <div className="summary-history-meta">
                  <span>#{idx + 1}</span>
                  <span>{formatApproxClockRange(sessionStartedAt, item.time_start, item.time_end)}</span>
                  <span>{formatRange(item.time_start, item.time_end)}</span>
                </div>
                <div>{item.content}</div>
              </article>
            ))
          )}
        </div>
      </div>
    </div>
  ) : null;

  if (compact) {
    return (
      <div className="workspace-tab-panel">
        <div className="workspace-tab-meta">
          <span className="summary-meta">covered {relativeRange} · updated {createdAt}</span>
          <span className="range">{range}</span>
        </div>
        {bodyJSX}
        {footerJSX}
        {modalJSX}
      </div>
    );
  }

  return (
    <section className={`panel ${className ?? ""}`.trim()} style={{ borderTop: `3px solid ${accent}` }}>
      <header>
        <div>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
          <p className="summary-meta">covered {relativeRange} · updated {createdAt}</p>
        </div>
        <div className="range">{range}</div>
      </header>
      {bodyJSX}
      {footerJSX}
      {modalJSX}
    </section>
  );
}

export default function App() {
  const [source, setSource] = useState("mixed");
  const [speechLanguage, setSpeechLanguage] = useState("auto");
  const [summaryLanguage, setSummaryLanguage] = useState("en");
  const [subtitleLanguage, setSubtitleLanguage] = useState("off");
  const [runtimeProfile, setRuntimeProfile] = useState("balanced");
  const [vocabulary, setVocabulary] = useState("");
  const [sessionEnabled, setSessionEnabled] = useState(false);
  const [lastExport, setLastExport] = useState<ExportMeetingResponse | null>(null);
  const [exportRoot, setExportRoot] = useState<string>("");
  const [exporting, setExporting] = useState(false);
  const [exportIntent, setExportIntent] = useState<"manual" | "restart" | null>(null);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);
  const [bookmarkFeedback, setBookmarkFeedback] = useState<string | null>(null);
  const [refreshingSummary, setRefreshingSummary] = useState<"rolling_summary" | "cumulative_meeting_summary" | "refined_transcript" | null>(null);
  const [refreshBaselineId, setRefreshBaselineId] = useState<string | null>(null);
  const [bilingualExport, setBilingualExport] = useState(false);
  const [restartPromptOpen, setRestartPromptOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [diagnosticsMode, setDiagnosticsMode] = useState<"simple" | "advanced">(
    () => (window.localStorage.getItem("meetingbro.diagnosticsMode") === "advanced" ? "advanced" : "simple"),
  );
  const [computeActivityHold, setComputeActivityHold] = useState({ cpu: false, gpu: false });
  const [vocabularyOpen, setVocabularyOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"summary" | "clean" | "board" | "notes">("summary");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [audioDevice, setAudioDevice] = useState("");
  const [loopbackDevice, setLoopbackDevice] = useState("");
  const [audioDevices, setAudioDevices] = useState<AudioDevices | null>(null);
  const { connected, state, meetingId, sessionStartedAt, elapsedSeconds, sessionStats, segments, previewSegment, previewSegments, isExperimentalPreview, previewDisplayText, previewIsStreaming, departingPreviewSegments, latestByType, historyByType, notes, lastError, lastErrorCode, saveNote, saveBookmark, applyVocabulary, exportMeeting, requestSummary, pauseSession, resumeSession, stopSession } =
    useSessionSocket({ enabled: sessionEnabled, source, speechLanguage, summaryLanguage, subtitleLanguage, runtimeProfile, audioDevice, loopbackDevice });

  const backend = useBackendStatus();

  // Load persisted device choices once, and fetch the device list when the
  // backend becomes ready.
  useEffect(() => {
    setAudioDevice(window.localStorage.getItem(AUDIO_DEVICE_KEY) ?? "");
    setLoopbackDevice(window.localStorage.getItem(LOOPBACK_DEVICE_KEY) ?? "");
  }, []);
  useEffect(() => {
    if (!backend.isReady) return;
    fetchAudioDevices().then(setAudioDevices).catch(() => setAudioDevices(null));
  }, [backend.isReady]);
  // First run: open Settings once so a new user lands on configuration instead
  // of guessing. Marking firstRunComplete touches no startup keys, so it won't
  // restart the backend.
  useEffect(() => {
    const bridge = window.meetingbro;
    if (!bridge?.getSettings || !backend.isReady) return;
    bridge.getSettings().then((s) => {
      if (!s.firstRunComplete) {
        setSettingsOpen(true);
        bridge.saveSettings?.({ firstRunComplete: true }).catch(() => {});
      }
    }).catch(() => {});
  }, [backend.isReady]);
  useEffect(() => {
    window.localStorage.setItem(AUDIO_DEVICE_KEY, audioDevice);
  }, [audioDevice]);
  useEffect(() => {
    window.localStorage.setItem(LOOPBACK_DEVICE_KEY, loopbackDevice);
  }, [loopbackDevice]);
  useEffect(() => {
    window.localStorage.setItem("meetingbro.diagnosticsMode", diagnosticsMode);
  }, [diagnosticsMode]);

  // Tell the Electron supervisor when a session is live so it won't restart the
  // backend (e.g. after a settings change) mid-recording.
  useEffect(() => {
    window.meetingbro?.setSessionActive?.(sessionEnabled);
  }, [sessionEnabled]);

  const rolling = latestByType.rolling_summary;
  const refined = latestByType.refined_transcript;
  const cumulative = latestByType.cumulative_meeting_summary;
  const chapterSnapshot = latestByType.chapter_list;
  const actionItemSnapshot = latestByType.action_item_list;
  const hasRetainedMeetingData = !sessionEnabled && !!meetingId && (
    segments.length > 0 ||
    notes.length > 0 ||
    Object.keys(latestByType).length > 0
  );

  useEffect(() => {
    setVocabulary(window.localStorage.getItem(VOCABULARY_STORAGE_KEY) ?? "");
  }, []);

  useEffect(() => {
    if (!actionFeedback) {
      return;
    }
    const timer = window.setTimeout(() => setActionFeedback(null), 3200);
    return () => window.clearTimeout(timer);
  }, [actionFeedback]);

  useEffect(() => {
    if (!bookmarkFeedback) {
      return;
    }
    const timer = window.setTimeout(() => setBookmarkFeedback(null), 2200);
    return () => window.clearTimeout(timer);
  }, [bookmarkFeedback]);

  useEffect(() => {
    if (!refreshingSummary) {
      return;
    }
    const latest = latestByType[refreshingSummary];
    if (!latest || latest.id === refreshBaselineId) {
      return;
    }
    setRefreshingSummary((current) => current === refreshingSummary ? null : current);
    setRefreshBaselineId(null);
  }, [latestByType, refreshBaselineId, refreshingSummary]);

  const visibleSegments = useMemo(() => segments.slice(-200), [segments]);
  const visibleBookmarks = useMemo(() => {
    if (visibleSegments.length === 0) {
      return [];
    }
    const firstVisibleStart = visibleSegments[0].start_time;
    const lastVisibleEnd = visibleSegments[visibleSegments.length - 1].end_time;
    const latestTimelineSeconds = Math.max(lastVisibleEnd, elapsedSeconds || 0);
    return notes
      .filter((note) => note.source_type === "bookmark" && note.time_seconds != null)
      .filter((note) => (note.time_seconds ?? 0) >= firstVisibleStart && (note.time_seconds ?? 0) <= latestTimelineSeconds)
      .sort((left, right) => (left.time_seconds ?? 0) - (right.time_seconds ?? 0));
  }, [elapsedSeconds, notes, visibleSegments]);
  const transcriptTimelineItems = useMemo(() => {
    const items: Array<
      | { type: "segment"; segment: typeof visibleSegments[number] }
      | { type: "bookmark"; note: typeof notes[number] }
    > = [];
    let bookmarkIndex = 0;

    for (const segment of visibleSegments) {
      while (
        bookmarkIndex < visibleBookmarks.length &&
        (visibleBookmarks[bookmarkIndex].time_seconds ?? 0) <= segment.end_time
      ) {
        items.push({ type: "bookmark", note: visibleBookmarks[bookmarkIndex] });
        bookmarkIndex += 1;
      }
      items.push({ type: "segment", segment });
    }

    while (bookmarkIndex < visibleBookmarks.length) {
      items.push({ type: "bookmark", note: visibleBookmarks[bookmarkIndex] });
      bookmarkIndex += 1;
    }

    return items;
  }, [notes, visibleBookmarks, visibleSegments]);
  const chapters = useMemo(() => chapterSnapshot ? parseSnapshotList(chapterSnapshot.content) : [], [chapterSnapshot]);
  const actionItems = useMemo(() => actionItemSnapshot ? parseSnapshotList(actionItemSnapshot.content) : [], [actionItemSnapshot]);
  const latestSegment = segments.at(-1);
  const [newSegmentOrders, setNewSegmentOrders] = useState<Record<string, number>>({});
  const segmentAnimationKnownIdsRef = useRef<Set<string>>(new Set());
  const latestVisualSegment = previewSegment ?? latestSegment;
  const handoffPreviewSegments = useMemo(
    () => [...departingPreviewSegments, ...previewSegments],
    [departingPreviewSegments, previewSegments],
  );
  const shouldShowPreviewStack = state === "running" || departingPreviewSegments.length > 0 || previewSegments.length > 0;
  const previewStackRevision = useMemo(
    () => handoffPreviewSegments.map((segment) => `${segment.id}:${segment.text}`).join("|"),
    [handoffPreviewSegments],
  );
  const latestSegmentRevision = latestSegment ? `${latestSegment.id}:${latestSegment.text}` : "";
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const knownIds = segmentAnimationKnownIdsRef.current;

    if (segments.length === 0) {
      knownIds.clear();
      setNewSegmentOrders({});
      return;
    }

    const newlyAdded = segments.filter((segment) => !knownIds.has(segment.id));
    segments.forEach((segment) => knownIds.add(segment.id));

    if (newlyAdded.length === 0) {
      return;
    }

    const addedOrders = Object.fromEntries(
      newlyAdded.map((segment, index) => [segment.id, index]),
    ) as Record<string, number>;
    const addedIds = new Set(newlyAdded.map((segment) => segment.id));
    setNewSegmentOrders((prev) => ({ ...prev, ...addedOrders }));
    const timer = window.setTimeout(() => {
      setNewSegmentOrders((prev) => {
        const next = { ...prev };
        addedIds.forEach((id) => {
          delete next[id];
        });
        return next;
      });
    }, 1100);
    return () => window.clearTimeout(timer);
  }, [segments]);

  useEffect(() => {
    if (!sessionEnabled) return;
    const timer = window.setInterval(() => setNowMs(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [sessionEnabled]);

  const audioClockSeconds = useMemo(() => {
    if (elapsedSeconds > 0) {
      return elapsedSeconds;
    }
    if (!sessionStartedAt) {
      return null;
    }
    const sessionStartMs = new Date(sessionStartedAt).getTime();
    if (Number.isNaN(sessionStartMs)) {
      return null;
    }
    return Math.max(0, (nowMs - sessionStartMs) / 1000);
  }, [elapsedSeconds, nowMs, sessionStartedAt]);

  const committedLagSeconds = useMemo(() => {
    if (!latestSegment || state !== "running" || audioClockSeconds == null) {
      return null;
    }
    const committedAt = latestSegment.emitted_at_elapsed_seconds;
    if (committedAt != null) {
      return Math.max(0, committedAt - latestSegment.end_time);
    }
    return Math.max(0, audioClockSeconds - latestSegment.end_time);
  }, [audioClockSeconds, latestSegment, state]);

  const liveGapSeconds = useMemo(() => {
    if (!latestVisualSegment || state !== "running" || audioClockSeconds == null) {
      return null;
    }
    return Math.max(0, audioClockSeconds - latestVisualSegment.end_time);
  }, [audioClockSeconds, latestVisualSegment, state]);

  const transcriptLagSeconds = liveGapSeconds ?? committedLagSeconds;

  const visibleTranscriptSeconds = latestSegment?.end_time ?? null;
  const backpressureAgeSeconds = useMemo(() => {
    const lastBackpressure = sessionStats?.last_backpressure_elapsed_seconds;
    if (lastBackpressure == null || elapsedSeconds <= 0) {
      return null;
    }
    return Math.max(0, elapsedSeconds - lastBackpressure);
  }, [elapsedSeconds, sessionStats?.last_backpressure_elapsed_seconds]);
  const retryTotal = sessionStats?.retry_windows_total ?? 0;
  const retryImproved = sessionStats?.retry_windows_improved ?? 0;
  const retryUnchanged = sessionStats?.retry_windows_unchanged ?? 0;
  const retryDiverged = sessionStats?.retry_windows_diverged ?? 0;
  const asrRealtimeFactor = sessionStats?.asr_realtime_factor ?? null;
  const asrWallSeconds = sessionStats?.asr_last_wall_seconds ?? null;
  const asrAudioSeconds = sessionStats?.asr_last_audio_seconds ?? null;
  const asrSafeguardActive = sessionStats?.asr_safeguard_active ?? false;
  const asrSafeguardEvents = sessionStats?.asr_safeguard_events ?? 0;
  const resourcePressureLevel = sessionStats?.resource_pressure_level ?? "normal";
  const resourceGovernorPolicy = sessionStats?.resource_governor_policy ?? "balanced";
  const resourceGovernorReason = sessionStats?.resource_governor_reason ?? null;
  const resourceGovernorSkips = sessionStats?.resource_governor_skips ?? 0;
  const activeRuntimeProfile = sessionStats?.runtime_profile ?? runtimeProfile;
  const summaryOnlyMode = activeRuntimeProfile === "summary_only";
  const activeHardwareProfile = sessionStats?.hardware_profile ?? "auto";
  const activeHardwareSummary = sessionStats?.hardware_summary ?? "detecting local CPU/GPU profile";
  const computeCpuActive = sessionStats?.compute_cpu_active ?? false;
  const computeCpuAvailable = sessionStats?.compute_cpu_available ?? true;
  const computeCpuConfigured = sessionStats?.compute_cpu_configured ?? false;
  const computeGpuActive = sessionStats?.compute_gpu_active ?? false;
  const computeGpuAvailable = sessionStats?.compute_gpu_available ?? false;
  const computeGpuConfigured = sessionStats?.compute_gpu_configured ?? false;
  const computeActivityLabel = sessionStats?.compute_activity_label ?? "idle";
  const cpuChipTitle = computeCpuActive
    ? computeActivityLabel
    : computeCpuConfigured
      ? "CPU is configured for MeetingBro and waiting for Qwen/CPU work"
      : computeCpuAvailable
        ? "CPU is available, but no CPU ASR task is currently assigned"
        : "CPU is not available";
  const gpuChipTitle = computeGpuActive
    ? computeActivityLabel
    : computeGpuConfigured
      ? "GPU is configured for MeetingBro and waiting for Whisper work"
      : computeGpuAvailable
        ? "GPU is available, but MeetingBro is currently configured to use CPU"
        : "GPU is not available to the current Python ASR runtime";
  const displayCpuActive = computeCpuActive || computeActivityHold.cpu;
  const displayGpuActive = computeGpuActive || computeActivityHold.gpu;
  const activeChunkSeconds = sessionStats?.audio_chunk_seconds ?? null;
  const activeAccumSeconds = sessionStats?.asr_accumulation_seconds ?? null;
  const languageLockEnabled = sessionStats?.language_lock_enabled ?? false;
  const weakRescueAttempts = sessionStats?.weak_rescue_attempts ?? 0;
  const weakRescueEmitted = sessionStats?.weak_rescue_emitted ?? 0;
  const weakRescueBufferSeconds = sessionStats?.weak_rescue_buffer_seconds ?? 0;
  const translationPendingCount = sessionStats?.translation_pending_count ?? 0;
  const summaryPendingCount = sessionStats?.summary_pending_count ?? 0;
  const translationTrimTotal = sessionStats?.translation_backlog_trim_total ?? 0;
  const audioDropTotal = sessionStats?.audio_drop_total ?? 0;
  const audioInputBacklogSeconds = sessionStats?.audio_input_backlog_seconds ?? 0;
  const audioInputQueueDropTotal = sessionStats?.audio_input_queue_drop_total ?? 0;
  const audioQueueWaitSeconds = sessionStats?.audio_queue_wait_seconds ?? null;
  const eventQueueDepth = sessionStats?.event_queue_depth ?? 0;
  const fastPreviewEnabled = sessionStats?.fast_preview_enabled ?? false;
  const fastPreviewAttempts = sessionStats?.fast_preview_attempts ?? 0;
  const fastPreviewEmitted = sessionStats?.fast_preview_emitted ?? 0;
  const fastPreviewSkipped = sessionStats?.fast_preview_skipped ?? 0;
  const fastPreviewRtf = sessionStats?.fast_preview_realtime_factor ?? null;
  const fastPreviewScheduleDelay = sessionStats?.fast_preview_schedule_delay_seconds ?? null;
  const snapshotConcatWall = sessionStats?.snapshot_concat_wall_seconds ?? null;
  const previewContinuedDuringFormal = sessionStats?.preview_continued_during_formal ?? 0;
  const previewStaleSuppressed = sessionStats?.preview_stale_suppressed ?? 0;
  const previewAlignmentCompared = sessionStats?.preview_alignment_compared ?? 0;
  const previewAlignmentSimilarityAvg = sessionStats?.preview_alignment_similarity_avg ?? null;
  const previewAlignmentSimilarityLast = sessionStats?.preview_alignment_similarity_last ?? null;
  const previewUnconfirmedAfterFormal = sessionStats?.preview_unconfirmed_after_formal ?? 0;
  const previewUnconfirmedLastText = sessionStats?.preview_unconfirmed_last_text ?? null;
  const qwenTargetedRetryAttempts = sessionStats?.qwen_targeted_retry_attempts ?? 0;
  const qwenTargetedRetryRecovered = sessionStats?.qwen_targeted_retry_recovered ?? 0;
  const qwenTargetedRetryFailed = sessionStats?.qwen_targeted_retry_failed ?? 0;
  const qwenTargetedRetrySkipped = sessionStats?.qwen_targeted_retry_skipped ?? 0;
  const qwenTargetedRetryLastReason = sessionStats?.qwen_targeted_retry_last_reason ?? null;
  const delayTone = getDelayTone(transcriptLagSeconds);
  const activeSource = sessionStats?.source ?? source;
  const mixedMicGain = sessionStats?.mixed_microphone_gain ?? null;
  const mixedSystemGain = sessionStats?.mixed_system_gain ?? null;
  const mixedEffectiveMicGain = sessionStats?.mixed_effective_microphone_gain ?? null;
  const mixedAutoBalanceEnabled = sessionStats?.mixed_auto_balance_enabled ?? null;
  const mixedGainTone = getGainTone(mixedEffectiveMicGain, mixedMicGain, activeSource);
  const mixedGainRatio =
    activeSource === "mixed"
      ? `mic ${formatGain(mixedEffectiveMicGain)} vs system ${formatGain(mixedSystemGain)}`
      : "—";
  const mixedGainHint = getMixedGainHint(mixedGainTone, activeSource);
  const activeSubtitleLanguage: "zh" | "en" | "de" | null = summaryOnlyMode || subtitleLanguage === "off"
    ? null
    : ((sessionStats?.live_translation_language ?? subtitleLanguage) as "zh" | "en" | "de");
  const visibleSubtitleRevision = useMemo(() => {
    if (activeSubtitleLanguage == null) {
      return 0;
    }
    return visibleSegments.reduce((revision, segment, index) => {
      const text = segment.translations[activeSubtitleLanguage]?.trim();
      if (!text) {
        return revision;
      }
      return revision + text.length + index + 1;
    }, 0);
  }, [activeSubtitleLanguage, visibleSegments]);
  const recentSpeechLanguages = useMemo(() => {
    const languages = new Set<string>();
    for (const segment of visibleSegments.slice(-24)) {
      if (segment.original_language !== "unknown") {
        languages.add(segment.original_language);
      }
    }
    return languages;
  }, [visibleSegments]);
  const latestSubtitleText = activeSubtitleLanguage && latestSegment
    ? latestSegment.translations[activeSubtitleLanguage]
    : null;
  const subtitleStatus = activeSubtitleLanguage == null
    ? "off"
    : !latestSegment
      ? "waiting"
      : latestSegment.original_language === activeSubtitleLanguage
        ? "same language"
        : latestSubtitleText && latestSubtitleText.trim()
          ? "translated"
          : "translating";
  const subtitleStatusDetail = activeSubtitleLanguage == null
    ? "subtitles disabled"
    : !latestSegment
      ? `target ${formatLanguageLabel(activeSubtitleLanguage)} · waiting for transcript`
      : latestSegment.original_language === activeSubtitleLanguage
        ? `latest segment already in ${formatLanguageLabel(activeSubtitleLanguage)}`
        : latestSubtitleText && latestSubtitleText.trim()
          ? `latest subtitle ready in ${formatLanguageLabel(activeSubtitleLanguage)}`
          : `target ${formatLanguageLabel(activeSubtitleLanguage)} · translation pending`;

  const systemAdvice = useMemo<SystemAdvice[]>(() => {
    if (!sessionEnabled || state !== "running") {
      return [{
        severity: "ok",
        title: "Ready to monitor",
        detail: "Start a session to receive live stability, accuracy, and latency advice.",
      }];
    }

    const advice: SystemAdvice[] = [];
    const realtimeGap = transcriptLagSeconds ?? 0;
    const rtf = asrRealtimeFactor ?? 0;
    const asrBusy = lastError?.includes("ASR has been processing") ?? false;

    if (asrBusy) {
      advice.push({
        severity: "warn",
        title: "ASR is currently busy",
        detail: lastError ?? "The backend is still processing the current audio window.",
        action: "Use Summary Only mode, switch to a smaller Whisper model, or turn subtitles off if lag continues.",
      });
    } else if (asrSafeguardActive || rtf >= 1.0 || backpressureAgeSeconds != null) {
      advice.push({
        severity: asrSafeguardActive || rtf >= 1.2 ? "danger" : "warn",
        title: "ASR is close to falling behind",
        detail: asrSafeguardActive
          ? (sessionStats?.asr_safeguard_reason ?? "The backend activated realtime protection.")
          : `Last ASR realtime factor is ${formatRtf(asrRealtimeFactor)}.`,
        action: "Use Summary Only mode, switch to a smaller Whisper model, or turn subtitles off if lag continues.",
      });
    } else if (realtimeGap >= 4.0 && activeRuntimeProfile !== "summary_only") {
      advice.push({
        severity: "warn",
        title: "Transcript is lagging the audio",
        detail: `Latest visible transcript is ${formatLagSeconds(realtimeGap)} behind the audio clock.`,
        action: "Try Summary Only mode if you can trade live captions for lower device load.",
      });
    }

    if (weakRescueAttempts >= 2 || weakRescueBufferSeconds >= 2.0) {
      advice.push({
        severity: weakRescueEmitted > 0 ? "warn" : "ok",
        title: "Quiet speech detected",
        detail: `Weak voice rescue attempted ${weakRescueAttempts} time(s), emitted ${weakRescueEmitted}.`,
        action: "Increase speaker volume, move closer to the microphone, or use Performance mode.",
      });
    }

    if (mixedGainTone === "warn" || mixedGainTone === "danger") {
      advice.push({
        severity: mixedGainTone,
        title: "Mixed audio balance may be uneven",
        detail: mixedGainHint,
        action: "Adjust microphone/system volume or keep Mixed mode auto-balance enabled.",
      });
    }

    if (translationPendingCount >= 4 || translationTrimTotal > 0) {
      advice.push({
        severity: translationTrimTotal > 0 ? "warn" : "ok",
        title: "Subtitle translation is busy",
        detail: `Pending translations ${translationPendingCount}, trims ${translationTrimTotal}.`,
        action: "If the machine is struggling, turn subtitles off or use Summary Only mode.",
      });
    }

    if (audioDropTotal > 0) {
      advice.push({
        severity: audioDropTotal >= 5 ? "danger" : "warn",
        title: "Audio chunks were dropped",
        detail: `${audioDropTotal} capture chunk(s) have been dropped.`,
        action: "Close other audio-heavy apps or switch source mode if drops continue.",
      });
    }

    if (audioInputBacklogSeconds >= 2.0 || audioInputQueueDropTotal > 0) {
      advice.push({
        severity: audioInputBacklogSeconds >= 5.0 || audioInputQueueDropTotal > 0 ? "warn" : "ok",
        title: "Audio input is buffered",
        detail: `Backlog ${formatLagSeconds(audioInputBacklogSeconds)}, queue drops ${audioInputQueueDropTotal}.`,
        action: "Use Summary Only mode or a smaller Whisper model until backlog stays near 0s.",
      });
    }

    if (fastPreviewEnabled && fastPreviewAttempts > 0 && fastPreviewEmitted === 0) {
      advice.push({
        severity: "warn",
        title: "Fast preview has not emitted text yet",
        detail: `Preview attempts ${fastPreviewAttempts}, skipped ${fastPreviewSkipped}.`,
        action: "This can happen on silence or if ASR is busy; if speech is present, try a smaller preview model later.",
      });
    }

    if (speechLanguage !== "auto" && recentSpeechLanguages.size >= 2) {
      advice.push({
        severity: "warn",
        title: "Multiple speech languages detected",
        detail: `Recent transcript contains ${Array.from(recentSpeechLanguages).join(", ")}.`,
        action: "Set Speech to Auto if the meeting is switching languages.",
      });
    }

    if (advice.length === 0) {
      advice.push({
        severity: "ok",
        title: "Realtime path looks healthy",
        detail: "No stability, latency, or accuracy warnings are active.",
        action: activeRuntimeProfile === "balanced"
          ? "Keep Balanced mode unless you need lower load or stronger accuracy bias."
          : `Current mode: ${formatRuntimeProfileLabel(activeRuntimeProfile)}.`,
      });
    }

    return advice.slice(0, 4);
  }, [
    activeRuntimeProfile,
    asrRealtimeFactor,
    asrSafeguardActive,
    audioDropTotal,
    audioInputBacklogSeconds,
    audioInputQueueDropTotal,
    backpressureAgeSeconds,
    fastPreviewAttempts,
    fastPreviewEmitted,
    fastPreviewEnabled,
    fastPreviewSkipped,
    languageLockEnabled,
    lastError,
    mixedGainHint,
    mixedGainTone,
    recentSpeechLanguages,
    sessionEnabled,
    sessionStats?.asr_safeguard_reason,
    speechLanguage,
    state,
    transcriptLagSeconds,
    translationPendingCount,
    translationTrimTotal,
    weakRescueAttempts,
    weakRescueBufferSeconds,
    weakRescueEmitted,
  ]);
  const systemAdviceTone = systemAdvice.some((item) => item.severity === "danger")
    ? "danger"
    : systemAdvice.some((item) => item.severity === "warn")
      ? "warn"
      : "ok";

  const transcriptBodyRef = useRef<HTMLDivElement>(null);
  const transcriptBottomRef = useRef<HTMLDivElement>(null);
  const transcriptAutoFollowRef = useRef(true);
  const transcriptProgrammaticScrollRef = useRef(false);
  const transcriptFollowFrameRef = useRef<number | null>(null);
  const [transcriptAutoFollow, setTranscriptAutoFollow] = useState(true);

  const scrollTranscriptToBottom = (behavior: ScrollBehavior = "auto") => {
    if (transcriptFollowFrameRef.current !== null) {
      window.cancelAnimationFrame(transcriptFollowFrameRef.current);
    }
    transcriptFollowFrameRef.current = window.requestAnimationFrame(() => {
      transcriptFollowFrameRef.current = null;
      const body = transcriptBodyRef.current;
      if (!body) return;
      transcriptProgrammaticScrollRef.current = true;
      if (behavior === "smooth") {
        body.scrollTo({ top: body.scrollHeight, behavior });
      } else {
        body.scrollTop = body.scrollHeight;
      }
      window.setTimeout(() => {
        const latestBody = transcriptBodyRef.current;
        if (latestBody && transcriptAutoFollowRef.current) {
          latestBody.scrollTop = latestBody.scrollHeight;
        }
        transcriptProgrammaticScrollRef.current = false;
      }, behavior === "smooth" ? 420 : 60);
    });
  };

  useLayoutEffect(() => {
    if (transcriptAutoFollowRef.current) {
      scrollTranscriptToBottom("auto");
    }
  }, [
    visibleSegments.length,
    latestSegmentRevision,
    visibleSubtitleRevision,
    previewSegment?.id,
    previewStackRevision,
    previewSegments.length,
    departingPreviewSegments.length,
    latestSegment?.id,
    latestSubtitleText,
    activeSubtitleLanguage,
  ]);

  useEffect(() => {
    return () => {
      if (transcriptFollowFrameRef.current !== null) {
        window.cancelAnimationFrame(transcriptFollowFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!computeCpuActive && !computeGpuActive) {
      return;
    }
    setComputeActivityHold((prev) => ({
      cpu: prev.cpu || computeCpuActive,
      gpu: prev.gpu || computeGpuActive,
    }));
    const timer = window.setTimeout(() => {
      setComputeActivityHold({ cpu: false, gpu: false });
    }, 1400);
    return () => window.clearTimeout(timer);
  }, [computeCpuActive, computeGpuActive]);

  const notesBodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (notesBodyRef.current) {
      notesBodyRef.current.scrollTop = notesBodyRef.current.scrollHeight;
    }
  }, [notes]);

  const handleSaveToNotes = async (snapshot: SummarySnapshot) => {
    await saveNote({
      content: snapshot.content,
      source_type: snapshot.summary_type,
      source_id: snapshot.id,
    });
  };

  const currentExportInput = (): ExportMeetingInput => ({
    source: activeSource,
    runtime_profile: activeRuntimeProfile,
    summary_language: summaryLanguage,
    subtitle_language: summaryOnlyMode ? "off" : subtitleLanguage,
    export_dir: exportRoot.trim() || undefined,
    bilingual: bilingualExport,
    target_language: bilingualExport ? ((summaryLanguage === "zh" || summaryLanguage === "en" || summaryLanguage === "de") ? summaryLanguage : undefined) : undefined,
  });

  const handleExportMeeting = async () => {
    setExportIntent("manual");
    setActionFeedback(null);
    setExporting(true);
    try {
      const result = await exportMeeting(currentExportInput());
      if (result) {
        setLastExport(result);
        setActionFeedback(`Export ready in ${result.export_dir}`);
      }
    } finally {
      setExporting(false);
      setExportIntent(null);
    }
  };

  const handleChooseExportFolder = async () => {
    const chooser = window.meetingbro?.selectExportDirectory;
    if (!chooser) {
      return;
    }
    const selected = await chooser(defaultExportFolderName());
    if (selected) {
      setExportRoot(selected);
    }
  };

  const handleRuntimeProfileChange = (value: string) => {
    setRuntimeProfile(value);
    if (value === "summary_only") {
      setSubtitleLanguage("off");
      setActiveTab("summary");
    }
  };

  const handleSaveVocabulary = () => {
    window.localStorage.setItem(VOCABULARY_STORAGE_KEY, vocabulary);
    applyVocabulary(vocabulary);
  };

  const handleStartSession = () => {
    if (hasRetainedMeetingData) {
      setRestartPromptOpen(true);
      return;
    }
    setLastExport(null);
    setSessionEnabled(true);
  };

  const handleRestartWithoutSave = () => {
    setRestartPromptOpen(false);
    setLastExport(null);
    setActionFeedback(null);
    setSessionEnabled(true);
  };

  const handleSaveAndRestart = async () => {
    setExportIntent("restart");
    setActionFeedback(null);
    setExporting(true);
    try {
      const result = await exportMeeting(currentExportInput());
      if (!result) {
        return;
      }
      setLastExport(result);
      setActionFeedback(`Saved to ${result.export_dir}. Starting a new meeting…`);
      setRestartPromptOpen(false);
      setSessionEnabled(true);
    } finally {
      setExporting(false);
      setExportIntent(null);
    }
  };

  const handleBookmark = async () => {
    const label = window.prompt("Bookmark label (optional)", "");
    if (label === null) {
      return;
    }
    await saveBookmark(label);
    setBookmarkFeedback(
      label.trim()
        ? `Bookmark saved at ${formatElapsedSeconds(audioClockSeconds ?? elapsedSeconds)} · ${label.trim()}`
        : `Bookmark saved at ${formatElapsedSeconds(audioClockSeconds ?? elapsedSeconds)}`,
    );
  };
  const handleRefreshSummary = (summaryType: "rolling_summary" | "cumulative_meeting_summary" | "refined_transcript") => {
    setRefreshBaselineId(latestByType[summaryType]?.id ?? null);
    setRefreshingSummary(summaryType);
    requestSummary(summaryType);
  };
  const vocabularyHint = sessionEnabled
    ? "Add keywords for better recognition. Save anytime, even during a meeting."
    : "Add keywords for better recognition. Save anytime before or during a meeting.";
  const vocabularyTermCount = useMemo(() => {
    return vocabulary
      .split(/[\n,，;；]+/)
      .map((term) => term.trim())
      .filter(Boolean).length;
  }, [vocabulary]);

  const transcriptEmptyMessage = !sessionEnabled
    ? "session stopped — press Start to begin capture"
    : state === "starting"
      ? "connecting to the live session…"
      : state === "paused"
        ? "session paused — press Resume to continue capture"
      : activeSource === "loopback" || activeSource === "system"
        ? "listening to system audio only — play meeting audio or switch to System + Microphone"
        : activeSource === "mic"
          ? "listening to microphone — speak to start the transcript"
          : "listening to system audio + microphone — speak or play meeting audio to start the transcript";

  return (
    <div className="app">
      {!backend.isReady && (
        <BackendGate
          state={backend.state}
          detail={backend.detail}
          logTail={backend.logTail}
          onRetry={backend.retry}
          onOpenLog={backend.openLog}
        />
      )}
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} sessionActive={sessionEnabled} />
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-row">
            <strong>MeetingBro</strong>
            <span className={`session-state state-${state}`}>{state}</span>
            {meetingId && <span className="meeting-id-pill">#{meetingId.slice(0, 8)}</span>}
            <span className={`compute-chip ${displayCpuActive ? "active" : computeCpuConfigured ? "configured" : computeCpuAvailable ? "available" : "idle"}`} title={cpuChipTitle}>
              <span className="compute-dot" aria-hidden="true" /> CPU
            </span>
            <span className={`compute-chip ${displayGpuActive ? "active" : computeGpuConfigured ? "configured" : computeGpuAvailable ? "available" : "idle"}`} title={gpuChipTitle}>
              <span className="compute-dot" aria-hidden="true" /> GPU
            </span>
          </div>
          <span className="app-tagline">Live captions ? AI notes ? clean meeting record</span>
        </div>
        <div className="header-controls" aria-label="Meeting controls">
          <div className="settings-controls" aria-label="Capture and language settings">
            <label className="control-pill">
              <span>Capture</span>
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="mic">Mic</option>
                <option value="loopback">System audio</option>
                <option value="mixed">System + Mic</option>
              </select>
            </label>
            {(source === "mic" || source === "mixed") && audioDevices && audioDevices.mics.length > 0 && (
              <label className="control-pill">
                <span>Mic device</span>
                <select value={audioDevice} onChange={(e) => setAudioDevice(e.target.value)}>
                  <option value="">Default</option>
                  {audioDevices.mics.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </label>
            )}
            {(source === "loopback" || source === "system") && audioDevices && audioDevices.loopbacks.length > 0 && (
              <label className="control-pill">
                <span>Output device</span>
                <select value={loopbackDevice} onChange={(e) => setLoopbackDevice(e.target.value)}>
                  <option value="">Default</option>
                  {audioDevices.loopbacks.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </label>
            )}
            <label className="control-pill">
              <span>Mode</span>
              <select value={runtimeProfile} onChange={(e) => handleRuntimeProfileChange(e.target.value)}>
                <option value="balanced">Balanced</option>
                <option value="performance">Performance</option>
                <option value="summary_only">Summary only</option>
              </select>
            </label>
            <label className="control-pill">
              <span>Speech</span>
              <select value={speechLanguage} onChange={(e) => setSpeechLanguage(e.target.value)}>
                <option value="auto">Auto</option>
                <option value="en">English</option>
                <option value="zh">Chinese</option>
                <option value="de">Deutsch</option>
              </select>
            </label>
            <label className="control-pill">
              <span>Notes</span>
              <select value={summaryLanguage} onChange={(e) => setSummaryLanguage(e.target.value)}>
                <option value="en">English</option>
                <option value="zh">Chinese</option>
                <option value="de">Deutsch</option>
              </select>
            </label>
            <label className="control-pill">
              <span>Captions</span>
              <select value={summaryOnlyMode ? "off" : subtitleLanguage} onChange={(e) => setSubtitleLanguage(e.target.value)} disabled={summaryOnlyMode}>
                <option value="off">Off</option>
                <option value="en">English</option>
                <option value="zh">Chinese</option>
                <option value="de">Deutsch</option>
              </select>
            </label>
            <button type="button" className="settings-gear" onClick={() => setSettingsOpen(true)} title="Settings" aria-label="Settings">
              ⚙
            </button>
          </div>
          <div className="session-controls" aria-label="Session actions">
            {!sessionEnabled ? (
              <button className="start-btn" onClick={handleStartSession} disabled={!backend.isReady}>Start</button>
            ) : (
              <>
                {state === "paused" ? (
                  <button className="resume-btn" onClick={resumeSession}>Resume</button>
                ) : (
                  <button className="pause-btn" onClick={pauseSession} disabled={state !== "running"}>
                    Pause
                  </button>
                )}
                <button
                  className="stop-btn"
                  onClick={() => {
                    stopSession();
                    setSessionEnabled(false);
                  }}
                >
                  Stop
                </button>
              </>
            )}
            <span className={`connection-pill ${connected ? "connected" : "disconnected"}`}>
              <span className={connected ? "dot ok" : "dot bad"} />
              {connected ? "Online" : "Offline"}
            </span>
          </div>
        </div>
      </header>

      {lastError && !isSystemHealthOnlyMessage(lastError) && (
        lastErrorCode === "llm_unavailable" || lastErrorCode === "translation_unavailable" ? (
          <div className="error error--actionable">
            <span>{lastError}</span>
            <button type="button" className="settings-btn" onClick={() => setSettingsOpen(true)}>Open Settings</button>
          </div>
        ) : (
          <div className="error">{lastError}</div>
        )
      )}

      <section className={`startup-settings vocabulary-dock${vocabularyOpen ? " open" : ""}`} aria-label="Names and terms settings">
        <button
          type="button"
          className="vocabulary-toggle"
          aria-expanded={vocabularyOpen}
          onClick={() => setVocabularyOpen((open) => !open)}
        >
          <span className="vocabulary-toggle-main">
            <strong>Names &amp; Terms</strong>
            <span className="vocabulary-toggle-note">Improve recognition for names, acronyms, and project terms</span>
          </span>
          <span className="vocabulary-count">
            {vocabularyTermCount === 0 ? "No terms" : `${vocabularyTermCount} ${vocabularyTermCount === 1 ? "term" : "terms"}`}
          </span>
          <span className={`toggle-caret${vocabularyOpen ? " open" : ""}`} aria-hidden="true">▼</span>
        </button>
        {vocabularyOpen && (
          <div className="startup-settings-panel">
            <p className="muted">{vocabularyHint}</p>
            <div className="startup-settings-controls">
              <textarea
                value={vocabulary}
                onChange={(e) => setVocabulary(e.target.value)}
                placeholder="Anthropic, MeetingBro, Libin Mao, faster-whisper, pyannote"
                rows={2}
                aria-label="Names, acronyms, or project terms"
              />
              <button type="button" className="secondary-action-btn" onClick={handleSaveVocabulary}>Save Terms</button>
            </div>
          </div>
        )}
      </section>

      <main className="grid">
        {summaryOnlyMode ? (
          <section className="panel transcript summary-only-panel">
            <header>
              <div>
                <h2>
                  Summary Only
                  {state === "running" && <span className="live-dot" aria-label="recording" />}
                </h2>
                <p className="subtitle">capturing in background · no live captions</p>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <button type="button" className="secondary-action-btn" onClick={handleBookmark} disabled={!connected || state !== "running"}>
                  Bookmark
                </button>
                <div className="range">{segments.length} segments</div>
              </div>
            </header>
            <div className="panel-body summary-only-body">
              <div className="summary-only-banner">
                <strong>Realtime captions are intentionally hidden in this mode.</strong>
                <p>
                  MeetingBro is still listening and collecting transcript segments, then updating Summary, Clean Notes,
                  and Board on their own cadence.
                </p>
              </div>
              <div className="summary-only-stats">
                <div className="summary-only-stat">
                  <span>Captured</span>
                  <strong>{segments.length} segments</strong>
                </div>
                <div className="summary-only-stat">
                  <span>Elapsed</span>
                  <strong>{formatElapsedSeconds(audioClockSeconds ?? elapsedSeconds)}</strong>
                </div>
                <div className="summary-only-stat">
                  <span>Latest summary</span>
                  <strong>{rolling?.created_at ? formatCreatedAt(rolling.created_at) : "waiting"}</strong>
                </div>
              </div>
              <p className="muted summary-only-tip">
                Tip: use the Refresh button in Summary or Clean Notes if you want an immediate update.
              </p>
            </div>
          </section>
        ) : <section className="panel transcript">
          <header>
            <div>
              <h2>
                Live Captions
                {state === "running" && <span className="live-dot" aria-label="recording" />}
              </h2>
              <p className="subtitle">confirmed · timestamped</p>
              <p className="transcript-lag-meta">
                <span className={`delay-pill delay-${delayTone}`}>{formatLagSeconds(transcriptLagSeconds)} behind</span>
                {committedLagSeconds != null ? ` · commit ${formatLagSeconds(committedLagSeconds)}` : ""}
                {previewSegment ? " · previewing" : ""}
              </p>
              {bookmarkFeedback && <p className="bookmark-feedback">{bookmarkFeedback}</p>}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button type="button" className="secondary-action-btn" onClick={handleBookmark} disabled={!connected || state !== "running"}>
                Bookmark
              </button>
              <div className="range">{segments.length} segments</div>
            </div>
          </header>
          <div
            ref={transcriptBodyRef}
            className="panel-body transcript-body"
            onScroll={(e) => {
              if (transcriptProgrammaticScrollRef.current) {
                return;
              }
              const follow = isNearBottom(e.currentTarget);
              transcriptAutoFollowRef.current = follow;
              setTranscriptAutoFollow(follow);
            }}
          >
            {visibleSegments.length === 0 && (
              <div className="muted">{transcriptEmptyMessage}</div>
            )}
            {transcriptTimelineItems.map((item) => {
              if (item.type === "bookmark") {
                return (
                  <div key={item.note.id} className="segment bookmark-marker">
                    <span className="ts">bookmark · {formatElapsedSeconds(item.note.time_seconds)}</span>
                    <div className="segment-text">
                      <span>{item.note.content.trim() || "Important moment"}</span>
                    </div>
                  </div>
                );
              }

              const s = item.segment;
              const confidenceTone = getConfidenceTone(s.confidence);
              const confidenceLabel = getConfidenceLabel(s.confidence);
              const qualityStyle: CSSProperties | undefined = s.quality === "low"
                ? { opacity: 0.55, fontStyle: "italic" }
                : s.quality === "uncertain"
                  ? { opacity: 0.7 }
                  : undefined;
              const subtitleText = activeSubtitleLanguage ? s.translations[activeSubtitleLanguage] : null;
              const showSubtitlePlaceholder =
                activeSubtitleLanguage != null &&
                s.original_language !== activeSubtitleLanguage &&
                !(subtitleText && subtitleText.trim());
              const segmentOrder = newSegmentOrders[s.id];
              const isNewSegment = segmentOrder !== undefined;
              const segmentDelayMs = isNewSegment ? segmentOrder * FORMAL_SEGMENT_STAGGER_MS : 0;
              const segmentStyle: StaggerStyle | undefined = isNewSegment
                ? {
                  ...qualityStyle,
                  "--segment-delay": `${segmentDelayMs}ms`,
                  "--segment-text-delay": `${segmentDelayMs + FORMAL_CELL_STAGGER_MS}ms`,
                }
                : qualityStyle;
              const segmentClassName = [
                "segment",
                isNewSegment ? "segment-new" : "",
                isNewSegment && hasPreviewHandoff(s, handoffPreviewSegments) ? "segment-promoted" : "",
                confidenceTone !== "stable" ? `segment-${confidenceTone}` : "",
              ].filter(Boolean).join(" ");
              return (
                <div key={s.id} className={segmentClassName} style={segmentStyle}>
                  <span className="ts">
                    {formatCreatedAt(s.created_at)}
                    {confidenceLabel && <em className={`confidence-flag confidence-${confidenceTone}`}>{confidenceLabel}</em>}
                  </span>
                  <div className="segment-text">
                    <span>{s.text}</span>
                    {subtitleText && subtitleText.trim() && subtitleText.trim() !== s.text.trim() && (
                      <span className="segment-subtitle">{subtitleText}</span>
                    )}
                    {showSubtitlePlaceholder && (
                      <span className="segment-subtitle segment-subtitle-pending" aria-label="translating subtitle">
                        <span className="subtitle-loading-dot" />
                        <span className="subtitle-loading-dot" />
                        <span className="subtitle-loading-dot" />
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
            {shouldShowPreviewStack && (
              <div className="preview-stack" aria-live="polite" aria-label="Live preview transcript">
                {departingPreviewSegments.map((segment, index) => {
                  const previewDelayMs = index * PREVIEW_ROW_STAGGER_MS;
                  const previewStyle: StaggerStyle = {
                    "--preview-status-delay": `${previewDelayMs}ms`,
                    "--preview-text-delay": `${previewDelayMs + PREVIEW_CELL_STAGGER_MS}ms`,
                  };
                  return (
                  <div key={`leaving-${segment.id}`} className="segment preview-segment preview-age-0 preview-leaving" style={previewStyle} aria-hidden="true">
                    <span className="preview-status">
                      <span className="preview-hearing">
                        <span className="preview-pulse-dot" aria-hidden="true" />
                        Listening
                      </span>
                    </span>
                    <span className="preview-segment-text">{segment.text}</span>
                  </div>
                  );
                })}
                {previewSegments.length > 0 && previewSegments.map((segment, index) => {
                  const isNewest = index === previewSegments.length - 1;
                  const age = previewSegments.length - index - 1;
                  const displayText = isNewest
                    ? previewDisplayText ?? (previewIsStreaming ? "" : segment.text)
                    : segment.text;
                  const showCursor = isNewest && !previewIsStreaming && previewDisplayText !== null;
                  const previewDelayMs = (departingPreviewSegments.length + index) * PREVIEW_ROW_STAGGER_MS;
                  const previewStyle: StaggerStyle = {
                    "--preview-status-delay": `${previewDelayMs}ms`,
                    "--preview-text-delay": `${previewDelayMs + PREVIEW_CELL_STAGGER_MS}ms`,
                  };
                  return (
                    <div key={segment.id} className={`segment preview-segment preview-age-${Math.min(age, 4)}`} style={previewStyle}>
                      <span className="preview-status">
                        <span className="preview-hearing">
                          <span className="preview-pulse-dot" aria-hidden="true" />
                          Listening
                        </span>
                        {isExperimentalPreview && (
                          <span
                            className="preview-experimental-badge"
                            title="Qwen3 fast preview · text is live and may still change"
                          >
                            Qwen · Live
                          </span>
                        )}
                      </span>
                      <span className="preview-segment-text">
                        {displayText}
                        {showCursor && <span className="preview-cursor" aria-hidden="true" />}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
            <div ref={transcriptBottomRef} className="transcript-bottom-anchor" />
            {!transcriptAutoFollow && (
              <button
                type="button"
                className="jump-latest-btn"
                onClick={() => {
                  transcriptAutoFollowRef.current = true;
                  setTranscriptAutoFollow(true);
                  scrollTranscriptToBottom("smooth");
                }}
              >
                Jump to latest
              </button>
            )}
          </div>
        </section>}

        <section className="panel workspace">
          <div className="workspace-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "summary"}
              className={`workspace-tab-btn${activeTab === "summary" ? " active" : ""}`}
              onClick={() => setActiveTab("summary")}
            >
              Summary
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "clean"}
              className={`workspace-tab-btn${activeTab === "clean" ? " active" : ""}`}
              onClick={() => setActiveTab("clean")}
            >
              Clean Notes
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "board"}
              className={`workspace-tab-btn${activeTab === "board" ? " active" : ""}`}
              onClick={() => setActiveTab("board")}
            >
              Board
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "notes"}
              className={`workspace-tab-btn${activeTab === "notes" ? " active" : ""}`}
              onClick={() => setActiveTab("notes")}
            >
              Notes
            </button>
          </div>
          <div className="workspace-content">
            {activeTab === "summary" && (
              <SummaryPanel
                compact
                title="AI Summary"
                subtitle="rolling AI digest of the last 3–5 minutes"
                snapshot={rolling}
                history={historyByType.rolling_summary ?? []}
                sessionStartedAt={sessionStartedAt}
                accent="#2563eb"
                onSaveToNotes={handleSaveToNotes}
                onRefresh={() => handleRefreshSummary("rolling_summary")}
                refreshDisabled={!connected || state !== "running" || segments.length === 0}
                refreshBusy={refreshingSummary === "rolling_summary"}
              />
            )}
            {activeTab === "clean" && (
              <SummaryPanel
                compact
                title="AI Clean Notes"
                subtitle="polished conversation record · Whisper + Qwen + LLM"
                snapshot={refined}
                history={historyByType.refined_transcript ?? []}
                sessionStartedAt={sessionStartedAt}
                accent="#7c3aed"
                onSaveToNotes={handleSaveToNotes}
                onRefresh={() => handleRefreshSummary("refined_transcript")}
                refreshDisabled={!connected || state !== "running" || segments.length === 0}
                refreshBusy={refreshingSummary === "refined_transcript"}
              />
            )}
            {activeTab === "board" && (
              <SummaryPanel
                compact
                title="Meeting Board"
                subtitle="live meeting state · decisions · actions · open questions"
                snapshot={cumulative}
                history={historyByType.cumulative_meeting_summary ?? []}
                sessionStartedAt={sessionStartedAt}
                accent="#0f766e"
                onSaveToNotes={handleSaveToNotes}
                onRefresh={() => handleRefreshSummary("cumulative_meeting_summary")}
                refreshDisabled={!connected || state !== "running" || segments.length === 0}
                refreshBusy={refreshingSummary === "cumulative_meeting_summary"}
              />
            )}
            {activeTab === "notes" && (
              <div className="workspace-tab-panel">
                <div ref={notesBodyRef} className="panel-body">
                  <div className="export-box">
                    <div>
                      <strong>Export meeting files</strong>
                      <p className="muted">Creates transcript.md, summary.md, and metadata.json in the chosen folder.</p>
                      <div className="export-target-row">
                        <input
                          type="text"
                          value={exportRoot}
                          onChange={(e) => setExportRoot(e.target.value)}
                          placeholder="Default: project exports/<timestamp> folder"
                          aria-label="Export folder"
                        />
                        {window.meetingbro?.selectExportDirectory && (
                          <button type="button" className="secondary-export-btn" onClick={handleChooseExportFolder} disabled={exporting}>
                            Browse…
                          </button>
                        )}
                      </div>
                      <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 10, fontSize: 12 }}>
                        <input
                          type="checkbox"
                          checked={bilingualExport}
                          onChange={(e) => setBilingualExport(e.target.checked)}
                        />
                        Bilingual transcript (original + translation)
                      </label>
                      {lastExport && (
                        <p className="export-path">
                          Latest export: <code>{lastExport.export_dir}</code>
                        </p>
                      )}
                      {(exporting || actionFeedback) && exportIntent !== "restart" && (
                        <p className={`action-feedback ${exporting ? "is-busy" : "is-success"}`}>
                          {exporting ? "Preparing transcript, summary, and metadata…" : actionFeedback}
                        </p>
                      )}
                    </div>
                    <button
                      type="button"
                      className={`primary-export-btn ${exporting && exportIntent === "manual" ? "button-busy" : ""}`.trim()}
                      onClick={handleExportMeeting}
                      disabled={!meetingId || exporting}
                      aria-busy={exporting && exportIntent === "manual"}
                    >
                      {exporting ? "Exporting…" : "Export Meeting"}
                    </button>
                  </div>
                  {chapterSnapshot && (
                    <div>
                      <strong>Chapters</strong>
                      {chapters.length === 0 ? (
                        <p className="muted">No chapters extracted.</p>
                      ) : (
                        <ul className="notes-list">
                          {chapters.map((chapter, index) => (
                            <li key={`chapter-${index}`}>
                              <div className="note-meta">
                                <span>{formatRange(Number(chapter.time_start ?? 0), Number(chapter.time_end ?? 0))}</span>
                                <span>{String(chapter.title ?? "Untitled chapter")}</span>
                              </div>
                              <div>{String(chapter.summary ?? "")}</div>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  {actionItemSnapshot && (
                    <div>
                      <strong>Action Items</strong>
                      {actionItems.length === 0 ? (
                        <p className="muted">No action items extracted.</p>
                      ) : (
                        <ul className="notes-list">
                          {actionItems.map((item, index) => (
                            <li key={`action-${index}`}>
                              <div>{String(item.text ?? "")}</div>
                              <div className="note-meta">
                                <span>{item.assignee ? `assignee: ${String(item.assignee)}` : "unassigned"}</span>
                                <span>{item.due ? `due: ${String(item.due)}` : "no due date"}</span>
                              </div>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  {notes.length === 0 ? (
                    <p className="muted">No saved notes yet. Use "Save to notes" on a summary panel.</p>
                  ) : (
                    <ul className="notes-list">
                      {notes.map((n) => (
                        <li key={n.id}>
                          <div className="note-meta">
                            <span>{n.source_type ?? "note"}</span>
                            <span className="muted">
                              {new Date(n.created_at).toLocaleTimeString()}
                            </span>
                          </div>
                          <div>{n.content}</div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      <div className="diagnostics-section">
        <button
          type="button"
          className="diagnostics-toggle"
          onClick={() => setDiagnosticsOpen((o) => !o)}
          aria-expanded={diagnosticsOpen}
          aria-controls="diagnostics-drawer"
        >
          <span className="diagnostics-toggle-label">System Health</span>
          <span className={`health-badge health-${systemAdviceTone}`}>
            {systemAdviceTone === "ok" ? "Healthy" : systemAdviceTone === "warn" ? "Check needed" : "Alert"}
          </span>
          {elapsedSeconds > 0 && (
            <span className="diagnostics-toggle-elapsed">{formatElapsedSeconds(elapsedSeconds)}</span>
          )}
          <span className={`toggle-caret${diagnosticsOpen ? " open" : ""}`} aria-hidden="true">▼</span>
        </button>
        {diagnosticsOpen && (
          <div id="diagnostics-drawer" className="diagnostics-drawer">
            <div className="diagnostics-body">
              <div className={`diagnostic-card advice-card advice-${systemAdviceTone}`}>
                <span className="diagnostic-label">System Advice</span>
                <div className="advice-list">
                  {systemAdvice.map((item) => (
                    <div key={`${item.title}-${item.detail}`} className={`advice-item advice-${item.severity}`}>
                      <strong>{item.title}</strong>
                      <span>{item.detail}</span>
                      {item.action && <em>{item.action}</em>}
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  className="diagnostics-mode-toggle"
                  onClick={() => setDiagnosticsMode((m) => (m === "simple" ? "advanced" : "simple"))}
                >
                  {diagnosticsMode === "simple" ? "Show advanced details" : "Hide advanced details"}
                </button>
              </div>
              {diagnosticsMode === "advanced" && (<>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Active Source</span>
                <strong className="diagnostic-value">{formatSourceLabel(activeSource)}</strong>
                <span className="diagnostic-note">
                  {activeSource === "mixed"
                    ? "capturing system audio and microphone"
                    : activeSource === "loopback" || activeSource === "system"
                      ? "capturing system output only"
                      : "capturing microphone only"}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Runtime Mode</span>
                <strong className="diagnostic-value">{formatRuntimeProfileLabel(activeRuntimeProfile)}</strong>
                <span className="diagnostic-note">
                  chunk {formatLagSeconds(activeChunkSeconds)} · ASR window {formatLagSeconds(activeAccumSeconds)} · lock {languageLockEnabled ? "on" : "off"}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Hardware Profile</span>
                <strong className="diagnostic-value">{activeHardwareProfile}</strong>
                <span className="diagnostic-note">{activeHardwareSummary}</span>
                {computeGpuAvailable && !computeGpuConfigured && (
                  <span className="diagnostic-note diagnostic-hint">GPU available but not assigned to ASR</span>
                )}
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Compute Activity</span>
                <strong className="diagnostic-value">{computeCpuActive || computeGpuActive ? "active" : "idle"}</strong>
                <span className="diagnostic-note">{computeActivityLabel}</span>
              </div>
              <div className={`diagnostic-card delay-card delay-${resourcePressureLevel === "critical" ? "danger" : resourcePressureLevel === "pressure" ? "warn" : "ok"}`}>
                <span className="diagnostic-label">Resource Governor</span>
                <strong className="diagnostic-value">{resourcePressureLevel}</strong>
                <span className="diagnostic-note">
                  {resourceGovernorPolicy} ? skips {resourceGovernorSkips}
                  {resourceGovernorReason ? ` ? ${resourceGovernorReason}` : ""}
                </span>
              </div>
              <div className={`diagnostic-card gain-card gain-${mixedGainTone}`}>
                <span className="diagnostic-label">Mixed Gain</span>
                <strong className="diagnostic-value">{mixedGainRatio}</strong>
                <span className="diagnostic-note">
                  {activeSource === "mixed"
                    ? `mic effective ${formatGain(mixedEffectiveMicGain)} · base ${formatGain(mixedMicGain)} · auto ${mixedAutoBalanceEnabled ? "on" : "off"}`
                    : mixedGainHint}
                </span>
                {activeSource === "mixed" && (
                  <span className="diagnostic-note diagnostic-hint">{mixedGainHint}</span>
                )}
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Subtitles</span>
                <strong className="diagnostic-value">{subtitleStatus}</strong>
                <span className="diagnostic-note">{subtitleStatusDetail}</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Audio Clock</span>
                <strong className="diagnostic-value">{formatElapsedSeconds(audioClockSeconds)}</strong>
                <span className="diagnostic-note">{elapsedSeconds > 0 ? "backend progress" : "frontend estimate"}</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Committed Transcript</span>
                <strong className="diagnostic-value">{formatElapsedSeconds(visibleTranscriptSeconds)}</strong>
                <span className="diagnostic-note">{latestSegment ? `last committed · ${formatRange(latestSegment.start_time, latestSegment.end_time)}` : "waiting for committed text"}</span>
              </div>
              <div className={`diagnostic-card delay-card delay-${delayTone}`}>
                <span className="diagnostic-label">Realtime Gap</span>
                <strong className="diagnostic-value">{formatLagSeconds(transcriptLagSeconds)}</strong>
                <span className="diagnostic-note">current audio clock minus latest visible transcript</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Commit Delay</span>
                <strong className="diagnostic-value">{formatLagSeconds(committedLagSeconds)}</strong>
                <span className="diagnostic-note">
                  {committedLagSeconds != null
                    ? "reference only · commit time minus segment end"
                    : "waiting for committed text"}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Retry Summary</span>
                <strong className="diagnostic-value">{retryTotal}</strong>
                <span className="diagnostic-note">improved {retryImproved} · unchanged {retryUnchanged} · diverged {retryDiverged}</span>
              </div>
              <div className={`diagnostic-card delay-card delay-${asrSafeguardActive ? "danger" : (asrRealtimeFactor != null && asrRealtimeFactor > 0.8 ? "warn" : "ok")}`}>
                <span className="diagnostic-label">ASR Realtime</span>
                <strong className="diagnostic-value">{formatRtf(asrRealtimeFactor)}</strong>
                <span className="diagnostic-note">
                  audio {formatLagSeconds(asrAudioSeconds)} · ASR {formatLagSeconds(asrWallSeconds)}
                </span>
              </div>
              <div className={`diagnostic-card delay-card delay-${fastPreviewRtf != null && fastPreviewRtf > 0.8 ? "warn" : "ok"}`}>
                <span className="diagnostic-label">Fast Preview</span>
                <strong className="diagnostic-value">{fastPreviewEnabled ? formatRtf(fastPreviewRtf) : "off"}</strong>
                <span className="diagnostic-note">
                  emitted {fastPreviewEmitted} · skipped {fastPreviewSkipped}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Realtime Internals</span>
                <strong className="diagnostic-value">q {eventQueueDepth}</strong>
                <span className="diagnostic-note">
                  audio wait {formatLagSeconds(audioQueueWaitSeconds)} · preview wake {formatLagSeconds(fastPreviewScheduleDelay)} · concat {formatLagSeconds(snapshotConcatWall)}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Preview Continued</span>
                <strong className="diagnostic-value">{previewContinuedDuringFormal}</strong>
                <span className="diagnostic-note">Qwen preview updates while formal text is pending</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Preview Stale Suppressed</span>
                <strong className="diagnostic-value">{previewStaleSuppressed}</strong>
                <span className="diagnostic-note">previews dropped (covered by formal)</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Preview Alignment</span>
                <strong className="diagnostic-value">
                  {previewAlignmentCompared > 0
                    ? `${(previewAlignmentSimilarityAvg ?? 0).toFixed(2)} avg`
                    : "—"}
                </strong>
                <span className="diagnostic-note">
                  compared {previewAlignmentCompared} · last {previewAlignmentSimilarityLast != null ? previewAlignmentSimilarityLast.toFixed(2) : "—"}
                </span>
              </div>
              <div className={`diagnostic-card delay-card delay-${previewUnconfirmedAfterFormal > 0 ? "warn" : "ok"}`}>
                <span className="diagnostic-label">Preview Unconfirmed</span>
                <strong className="diagnostic-value">{previewUnconfirmedAfterFormal}</strong>
                <span className="diagnostic-note">
                  {previewUnconfirmedLastText ? `last: ${previewUnconfirmedLastText}` : "no preview has been passed by formal without overlap"}
                </span>
              </div>
              <div className={`diagnostic-card delay-card delay-${qwenTargetedRetryFailed > 0 ? "warn" : "ok"}`}>
                <span className="diagnostic-label">Qwen Recovery</span>
                <strong className="diagnostic-value">{qwenTargetedRetryRecovered}</strong>
                <span className="diagnostic-note">
                  attempts {qwenTargetedRetryAttempts} · failed {qwenTargetedRetryFailed} · skipped {qwenTargetedRetrySkipped}
                  {qwenTargetedRetryLastReason ? ` · ${qwenTargetedRetryLastReason}` : ""}
                </span>
              </div>
              <div className={`diagnostic-card delay-card delay-${asrSafeguardActive ? "danger" : "ok"}`}>
                <span className="diagnostic-label">Realtime Safeguard</span>
                <strong className="diagnostic-value">{asrSafeguardActive ? "active" : "clear"}</strong>
                <span className="diagnostic-note">
                  {asrSafeguardActive
                    ? (sessionStats?.asr_safeguard_reason ?? "protecting realtime path")
                    : `events ${asrSafeguardEvents}`}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Weak Voice Rescue</span>
                <strong className="diagnostic-value">{weakRescueAttempts}</strong>
                <span className="diagnostic-note">
                  emitted {weakRescueEmitted} · buffer {formatLagSeconds(weakRescueBufferSeconds)}
                </span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Background Queue</span>
                <strong className="diagnostic-value">{translationPendingCount + summaryPendingCount}</strong>
                <span className="diagnostic-note">
                  translations {translationPendingCount} · summaries {summaryPendingCount} · trims {translationTrimTotal}
                </span>
              </div>
              <div className={`diagnostic-card delay-card delay-${audioInputBacklogSeconds >= 5 ? "warn" : "ok"}`}>
                <span className="diagnostic-label">Audio Backlog</span>
                <strong className="diagnostic-value">{formatLagSeconds(audioInputBacklogSeconds)}</strong>
                <span className="diagnostic-note">input queue drops {audioInputQueueDropTotal}</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Audio Drops</span>
                <strong className="diagnostic-value">{audioDropTotal}</strong>
                <span className="diagnostic-note">capture queue dropped chunks</span>
              </div>
              <div className="diagnostic-card">
                <span className="diagnostic-label">Backpressure</span>
                <strong className="diagnostic-value">{backpressureAgeSeconds != null ? "recent" : "clear"}</strong>
                <span className="diagnostic-note">
                  {backpressureAgeSeconds != null
                    ? `last seen ${formatLagSeconds(backpressureAgeSeconds)} ago at ${formatElapsedSeconds(sessionStats?.last_backpressure_elapsed_seconds)}`
                    : "no recent slow-ASR flush"}
                </span>
              </div>
              </>)}
            </div>
          </div>
        )}
      </div>

      {restartPromptOpen && (
        <div className="confirm-overlay" role="dialog" aria-modal="true" aria-labelledby="restart-dialog-title">
          <div className="confirm-dialog">
            <h3 id="restart-dialog-title">Start a new meeting?</h3>
            <p>
              Starting again will clear the current transcript, summaries, bookmarks, and notes from the screen.
            </p>
            <p className="muted">
              Save this meeting first if you want to keep an export copy before opening a new session.
            </p>
            <div className="confirm-actions">
              <button type="button" className="secondary-action-btn" onClick={() => setRestartPromptOpen(false)} disabled={exporting}>
                Cancel
              </button>
              <button type="button" className="danger-action-btn" onClick={handleRestartWithoutSave} disabled={exporting}>
                Start Without Saving
              </button>
              <button type="button" className={`primary-export-btn ${exporting && exportIntent === "restart" ? "button-busy" : ""}`.trim()} onClick={handleSaveAndRestart} disabled={exporting} aria-busy={exporting && exportIntent === "restart"}>
                {exporting ? "Saving…" : "Save and Start New"}
              </button>
            </div>
            {(exporting || actionFeedback) && (
              <p className={`confirm-status ${exporting ? "is-busy" : "is-success"}`}>
                {exporting ? "Saving the current meeting export before opening a new session…" : actionFeedback}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
