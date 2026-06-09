const { app, BrowserWindow, Menu, dialog, ipcMain, shell, screen } = require("electron");
const path = require("node:path");

const { BackendSupervisor } = require("./backend.cjs");
const settingsStore = require("./settings-store.cjs");

const isDev = !app.isPackaged;

let mainWindow = null;
let supervisor = null;
let sessionActive = false; // renderer reports this so we don't restart mid-recording

function broadcastBackendStatus(info) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("meetingbro:backend-status", info);
  }
}

async function startBackend() {
  const settings = settingsStore.readSettings();
  await supervisor.start({
    env: settingsStore.settingsToEnv(settings),
    port: settings.backendPort,
  });
}

function createWindow() {
  const { width: workWidth, height: workHeight } = screen.getPrimaryDisplay().workAreaSize;
  const width = Math.min(1680, Math.max(1280, Math.floor(workWidth * 0.92)));
  const height = Math.min(1000, Math.max(820, Math.floor(workHeight * 0.9)));

  const win = new BrowserWindow({
    width,
    height,
    minWidth: 1280,
    minHeight: 780,
    center: true,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.setMenu(null);
  win.setMenuBarVisibility(false);
  mainWindow = win;

  if (isDev) {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);

  supervisor = new BackendSupervisor({ onStatus: broadcastBackendStatus });

  // Synchronous so preload can expose the configured backend URLs at load time.
  ipcMain.on("meetingbro:get-backend-bases", (event) => {
    const port = settingsStore.readSettings().backendPort || 8765;
    event.returnValue = { httpBase: `http://127.0.0.1:${port}`, wsBase: `ws://127.0.0.1:${port}` };
  });

  ipcMain.handle("meetingbro:select-export-directory", async (_event, suggestedName) => {
    const defaultName = suggestedName || `meetingbro_export_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
    const result = await dialog.showSaveDialog({
      title: "Export meeting as folder",
      buttonLabel: "Use This Folder",
      defaultPath: path.join(app.getPath("documents"), defaultName),
      properties: ["createDirectory", "showOverwriteConfirmation"],
    });
    if (result.canceled || !result.filePath) {
      return null;
    }
    return result.filePath;
  });

  // ── Backend lifecycle / status ─────────────────────────────────────────────
  ipcMain.handle("meetingbro:get-backend-info", () => supervisor.info());
  ipcMain.handle("meetingbro:retry-backend", async () => {
    await startBackend();
    return supervisor.info();
  });
  ipcMain.handle("meetingbro:open-backend-log", () => {
    const logPath = path.join(app.getPath("userData"), "logs", "backend.log");
    shell.showItemInFolder(logPath);
    return logPath;
  });
  ipcMain.on("meetingbro:set-session-active", (_event, active) => {
    sessionActive = Boolean(active);
  });

  // ── Settings ────────────────────────────────────────────────────────────────
  ipcMain.handle("meetingbro:get-settings", () => settingsStore.readSettings());
  ipcMain.handle("meetingbro:save-settings", async (_event, partial) => {
    const before = settingsStore.startupSignature(settingsStore.readSettings());
    const saved = settingsStore.writeSettings(partial || {});
    const after = settingsStore.startupSignature(saved);
    const startupChanged = before !== after;
    let restarted = false;
    if (startupChanged && !sessionActive) {
      await supervisor.restart({ env: settingsStore.settingsToEnv(saved), port: saved.backendPort });
      restarted = true;
    }
    return { settings: saved, startupChanged, restarted, backend: supervisor.info() };
  });

  createWindow();
  startBackend();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", async (event) => {
  if (supervisor && !supervisor.quitting) {
    event.preventDefault();
    await supervisor.stop();
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
