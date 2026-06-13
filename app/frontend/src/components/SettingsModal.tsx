import { useEffect, useState } from "react";
import type { AppSettings } from "../meetingbro-bridge";
import { fetchConfig, testLlm } from "../system/backendApi";
import type { BackendConfig, TestLlmResult } from "../system/backendApi";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  sessionActive: boolean;
}

const PROVIDER_PRESETS: Record<string, { baseUrl: string; model: string }> = {
  OpenAI: { baseUrl: "https://api.openai.com/v1", model: "gpt-4o-mini" },
  LongCat: { baseUrl: "https://api.longcat.chat/openai", model: "LongCat-Flash-Chat" },
};

const WHISPER_SIZES = ["auto", "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"];
// Large tiers are slow on CPU; surface a GPU-recommended hint when no CUDA is available.
const GPU_RECOMMENDED_SIZES = new Set(["large-v2", "large-v3", "large-v3-turbo"]);

export function SettingsModal({ open, onClose, sessionActive }: SettingsModalProps) {
  const bridge = typeof window !== "undefined" ? window.meetingbro : undefined;
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<TestLlmResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveNote, setSaveNote] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setSaveNote(null);
    setTestResult(null);
    bridge?.getSettings?.().then(setSettings).catch(() => {});
    fetchConfig().then(setConfig).catch(() => setConfig(null));
  }, [open, bridge]);

  if (!open) return null;

  const hasBridge = Boolean(bridge?.saveSettings);

  const update = (patch: Partial<AppSettings>) => setSettings((prev) => (prev ? { ...prev, ...patch } : prev));
  const updateLlm = (patch: Partial<AppSettings["llm"]>) =>
    setSettings((prev) => (prev ? { ...prev, llm: { ...prev.llm, ...patch } } : prev));

  const applyProvider = (name: string) => {
    const preset = PROVIDER_PRESETS[name];
    if (preset) updateLlm({ baseUrl: preset.baseUrl, model: preset.model });
  };

  const onTest = async () => {
    if (!settings) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testLlm({ api_key: settings.llm.apiKey, base_url: settings.llm.baseUrl, model: settings.llm.model });
      setTestResult(result);
    } catch (err) {
      setTestResult({ ok: false, error: (err as Error).message });
    } finally {
      setTesting(false);
    }
  };

  const onSave = async () => {
    if (!settings || !bridge?.saveSettings) return;
    setSaving(true);
    setSaveNote(null);
    try {
      const res = await bridge.saveSettings(settings);
      if (res.restarted) setSaveNote("Saved. Restarting the backend to apply changes…");
      else if (res.startupChanged && sessionActive) setSaveNote("Saved. Changes apply after the current session ends and the backend restarts.");
      else setSaveNote("Saved.");
    } catch (err) {
      setSaveNote(`Save failed: ${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  const recommended = config?.hardware.recommended_whisper_size;
  const cudaAvailable = Boolean(config?.cuda_available);
  const showGpuSizeHint = Boolean(settings && GPU_RECOMMENDED_SIZES.has(settings.whisperSize) && !cudaAvailable);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-label="Settings" onClick={onClose}>
      <div className="modal modal--settings" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2>Settings</h2>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">×</button>
        </div>

        {!hasBridge && (
          <p className="modal__notice">In-app settings are available in the desktop app. In a browser, configure the backend via its .env file.</p>
        )}

        {settings && (
          <div className="modal__body">
            <section className="settings-section">
              <h3>AI summaries (optional)</h3>
              <p className="settings-hint">Leave blank to use offline keyword summaries. Add an OpenAI-compatible API key for higher-quality notes.</p>
              <label className="settings-field">
                <span>Provider preset</span>
                <select onChange={(e) => applyProvider(e.target.value)} defaultValue="">
                  <option value="" disabled>Choose…</option>
                  {Object.keys(PROVIDER_PRESETS).map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <label className="settings-field">
                <span>API key</span>
                <input type="password" value={settings.llm.apiKey} placeholder={config?.llm_key_masked ?? "sk-…"} onChange={(e) => updateLlm({ apiKey: e.target.value })} />
              </label>
              <label className="settings-field">
                <span>Base URL</span>
                <input type="text" value={settings.llm.baseUrl} onChange={(e) => updateLlm({ baseUrl: e.target.value })} />
              </label>
              <label className="settings-field">
                <span>Model</span>
                <input type="text" value={settings.llm.model} onChange={(e) => updateLlm({ model: e.target.value })} />
              </label>
              <div className="settings-actions-inline">
                <button type="button" className="settings-btn" onClick={onTest} disabled={testing || !settings.llm.apiKey}>
                  {testing ? "Testing…" : "Test connection"}
                </button>
                {testResult && (
                  <span className={`settings-test ${testResult.ok ? "is-ok" : "is-error"}`}>
                    {testResult.ok ? `✓ Connected (${testResult.latency_ms} ms)` : `✗ ${testResult.error}`}
                  </span>
                )}
              </div>
            </section>

            <section className="settings-section">
              <h3>Speech model</h3>
              <label className="settings-field">
                <span>Model size</span>
                <select value={settings.whisperSize} onChange={(e) => update({ whisperSize: e.target.value })}>
                  {WHISPER_SIZES.map((size) => (
                    <option key={size} value={size}>
                      {size === "auto" && recommended ? `Auto (recommended: ${recommended})` : size}
                    </option>
                  ))}
                </select>
              </label>
              {showGpuSizeHint && (
                <p className="settings-hint">
                  {settings?.whisperSize === "large-v3-turbo"
                    ? "large-v3-turbo is fastest among the large models but still GPU-recommended — on CPU it runs well below real time."
                    : "Large models are GPU-recommended — on CPU they run well below real time. Try small/medium, or large-v3-turbo if you need top accuracy."}
                </p>
              )}
              <label className="settings-field">
                <span>Compute device</span>
                <select value={settings.whisperDevice} onChange={(e) => update({ whisperDevice: e.target.value })}>
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  {config?.cuda_available && <option value="cuda">GPU (CUDA)</option>}
                </select>
              </label>
              <label className="settings-field settings-field--inline">
                <input type="checkbox" checked={settings.previewBackend === "qwen3"} onChange={(e) => update({ previewBackend: e.target.checked ? "qwen3" : "shared" })} />
                <span>Fast live captions (Qwen3 preview) — needs the Qwen3 model downloaded</span>
              </label>
            </section>
          </div>
        )}

        <div className="modal__footer">
          {saveNote && <span className="settings-save-note">{saveNote}</span>}
          {sessionActive && <span className="settings-hint">Model/key changes apply after the current session ends.</span>}
          <button type="button" className="settings-btn" onClick={onClose}>Close</button>
          <button type="button" className="settings-btn settings-btn--primary" onClick={onSave} disabled={!hasBridge || saving || !settings}>
            {saving ? "Saving…" : "Save & apply"}
          </button>
        </div>
      </div>
    </div>
  );
}
