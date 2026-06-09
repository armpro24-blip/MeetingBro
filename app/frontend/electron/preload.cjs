const { contextBridge, ipcRenderer } = require("electron");

// Resolve the backend base URLs synchronously at load so the existing
// useSessionSocket code (which reads window.meetingbro.backendHttp/backendWs)
// keeps working even when the user configured a non-default port.
let bases = { httpBase: "http://127.0.0.1:8765", wsBase: "ws://127.0.0.1:8765" };
try {
  const resolved = ipcRenderer.sendSync("meetingbro:get-backend-bases");
  if (resolved && resolved.httpBase && resolved.wsBase) bases = resolved;
} catch {
  /* fall back to defaults */
}

contextBridge.exposeInMainWorld("meetingbro", {
  backendHttp: bases.httpBase,
  backendWs: bases.wsBase,
  selectExportDirectory: (suggestedName) => ipcRenderer.invoke("meetingbro:select-export-directory", suggestedName),

  // Backend lifecycle / status
  getBackendInfo: () => ipcRenderer.invoke("meetingbro:get-backend-info"),
  onBackendStatus: (cb) => {
    const listener = (_event, info) => cb(info);
    ipcRenderer.on("meetingbro:backend-status", listener);
    return () => ipcRenderer.removeListener("meetingbro:backend-status", listener);
  },
  retryBackend: () => ipcRenderer.invoke("meetingbro:retry-backend"),
  openBackendLog: () => ipcRenderer.invoke("meetingbro:open-backend-log"),
  setSessionActive: (active) => ipcRenderer.send("meetingbro:set-session-active", active),

  // Settings
  getSettings: () => ipcRenderer.invoke("meetingbro:get-settings"),
  saveSettings: (partial) => ipcRenderer.invoke("meetingbro:save-settings", partial),
});
