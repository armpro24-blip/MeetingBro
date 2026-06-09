// Shape of the `window.meetingbro` bridge exposed by electron/preload.cjs.
// Single source of truth so the session hook and UI components agree.

export type BackendState =
  | "idle"
  | "locating"
  | "starting"
  | "downloading-model"
  | "ready"
  | "failed"
  | "stopped";

export interface BackendInfo {
  state: BackendState;
  detail: string | null;
  httpBase: string;
  wsBase: string;
  logTail: string[];
}

export interface LlmSettings {
  apiKey: string;
  baseUrl: string;
  model: string;
}

export interface AppSettings {
  firstRunComplete: boolean;
  backendPort: number;
  llm: LlmSettings;
  whisperSize: string;
  whisperDevice: string;
  previewBackend: "qwen3" | "shared";
}

export interface SaveSettingsResult {
  settings: AppSettings;
  startupChanged: boolean;
  restarted: boolean;
  backend: BackendInfo;
}

export interface MeetingBroBridge {
  backendHttp: string;
  backendWs: string;
  selectExportDirectory?: (suggestedName?: string) => Promise<string | null>;

  getBackendInfo: () => Promise<BackendInfo>;
  onBackendStatus: (cb: (info: BackendInfo) => void) => () => void;
  retryBackend: () => Promise<BackendInfo>;
  openBackendLog: () => Promise<string>;
  setSessionActive: (active: boolean) => void;

  getSettings: () => Promise<AppSettings>;
  saveSettings: (partial: Partial<AppSettings>) => Promise<SaveSettingsResult>;
}

declare global {
  interface Window {
    meetingbro?: MeetingBroBridge;
  }
}
