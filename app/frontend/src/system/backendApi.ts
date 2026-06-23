// Thin fetch wrappers over the backend's read-only/validation endpoints.

import type { ExportMeetingResponse, Note, SummarySnapshot, TranscriptSegment } from "../types";

function httpBase(): string {
  return window.meetingbro?.backendHttp ?? "http://127.0.0.1:8765";
}

export interface AudioDevice {
  id: string;
  name: string;
  default?: boolean;
}

export interface AudioDevices {
  mics: AudioDevice[];
  loopbacks: AudioDevice[];
  system_default: AudioDevice | null;
}

export interface BackendConfig {
  whisper_size: string;
  whisper_device: string;
  whisper_compute_type: string;
  preview_backend: string;
  port: number;
  cuda_available: boolean;
  llm_configured: boolean;
  llm_key_masked: string | null;
  llm_model: string | null;
  llm_base_url: string | null;
  hardware: {
    label: string;
    summary: string | null;
    recommended_whisper_size: string;
    recommended_whisper_device: string;
    recommended_runtime_profile: string;
  };
}

export interface TestLlmResult {
  ok: boolean;
  model?: string;
  latency_ms?: number;
  error?: string;
}

export async function fetchAudioDevices(): Promise<AudioDevices> {
  const res = await fetch(`${httpBase()}/audio/devices`);
  if (!res.ok) throw new Error(`audio/devices ${res.status}`);
  return res.json();
}

export async function fetchConfig(): Promise<BackendConfig> {
  const res = await fetch(`${httpBase()}/config`);
  if (!res.ok) throw new Error(`config ${res.status}`);
  return res.json();
}

export interface MeetingListItem {
  id: string;
  started_at: string;
  ended_at: string | null;
  preferred_summary_language: string;
  segment_count: number;
  title: string | null;
}

export async function fetchMeetings(limit = 100): Promise<MeetingListItem[]> {
  const res = await fetch(`${httpBase()}/meetings?limit=${limit}`);
  if (!res.ok) throw new Error(`meetings ${res.status}`);
  return res.json();
}

export async function fetchMeetingTranscript(id: string): Promise<TranscriptSegment[]> {
  const res = await fetch(`${httpBase()}/meetings/${id}/transcript`);
  if (!res.ok) throw new Error(`transcript ${res.status}`);
  return res.json();
}

export async function fetchMeetingSummaries(id: string): Promise<SummarySnapshot[]> {
  const res = await fetch(`${httpBase()}/meetings/${id}/summaries`);
  if (!res.ok) throw new Error(`summaries ${res.status}`);
  return res.json();
}

export async function fetchMeetingNotes(id: string): Promise<Note[]> {
  const res = await fetch(`${httpBase()}/meetings/${id}/notes`);
  if (!res.ok) throw new Error(`notes ${res.status}`);
  return res.json();
}

export async function exportMeetingById(id: string): Promise<ExportMeetingResponse> {
  const res = await fetch(`${httpBase()}/meetings/${id}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`export ${res.status}`);
  return res.json();
}

export async function deleteMeeting(id: string): Promise<void> {
  const res = await fetch(`${httpBase()}/meetings/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`delete ${res.status}`);
}

export async function testLlm(input: { api_key: string; base_url?: string; model?: string }): Promise<TestLlmResult> {
  const res = await fetch(`${httpBase()}/config/test-llm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
  return res.json();
}
