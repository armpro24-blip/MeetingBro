// Supervises the Python backend so non-technical users never open a terminal.
//
// Ported from scripts/start.ps1: locate the venv python, optionally attach to an
// already-running backend, otherwise spawn `python -m meetingbro.main`, poll
// /health until ready, and kill the process tree on quit. Status transitions are
// pushed to the renderer so the UI can show a friendly startup / error screen.

const { app } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const HEALTH_POLL_INTERVAL_MS = 500;
const LOG_TAIL_LINES = 200;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class BackendSupervisor {
  constructor({ onStatus } = {}) {
    this.onStatus = typeof onStatus === "function" ? onStatus : () => {};
    this.child = null;
    this.childExited = false;
    this.attached = false; // true when a foreign backend was already running
    this.quitting = false;
    this.port = 8765;
    this.state = "idle";
    this.detail = null;
    this.logTail = [];
    this.logStream = null;
    this._pollToken = 0;
  }

  httpBase() {
    return `http://127.0.0.1:${this.port}`;
  }
  wsBase() {
    return `ws://127.0.0.1:${this.port}`;
  }

  info() {
    return { state: this.state, detail: this.detail, httpBase: this.httpBase(), wsBase: this.wsBase(), logTail: this.logTail.slice(-40) };
  }

  _emit(state, detail) {
    this.state = state;
    this.detail = detail ?? null;
    this.onStatus(this.info());
  }

  _appendLog(line) {
    this.logTail.push(line);
    if (this.logTail.length > LOG_TAIL_LINES) this.logTail.shift();
    if (this.logStream) {
      try {
        this.logStream.write(line + "\n");
      } catch {
        /* ignore log write errors */
      }
    }
  }

  // Resolve repo paths relative to this file (app/frontend/electron/).
  _resolvePaths() {
    const repoRoot = path.resolve(__dirname, "..", "..", "..");
    const backendDir = path.join(repoRoot, "app", "backend");
    const venvPython =
      process.platform === "win32"
        ? path.join(backendDir, ".venv", "Scripts", "python.exe")
        : path.join(backendDir, ".venv", "bin", "python");
    return { backendDir, venvPython };
  }

  _openLogStream() {
    try {
      const logDir = path.join(app.getPath("userData"), "logs");
      fs.mkdirSync(logDir, { recursive: true });
      this.logStream = fs.createWriteStream(path.join(logDir, "backend.log"), { flags: "a" });
    } catch (err) {
      console.warn("[backend] could not open log file:", err.message);
    }
  }

  // GET /health with a short timeout. Returns the parsed body or null.
  async _probeHealth(timeoutMs = 1500) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${this.httpBase()}/health`, { signal: controller.signal });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  // Start (or attach to) the backend. `env` are the MEETINGBRO_* overrides.
  async start({ env = {}, port = 8765 } = {}) {
    this.port = Number(port) || 8765;
    this.quitting = false;
    this.childExited = false;
    this.attached = false;
    this._openLogStream();

    this._emit("locating");

    // If a healthy backend already answers on this port (e.g. the user ran
    // scripts/start.ps1), attach to it instead of spawning a duplicate.
    const existing = await this._probeHealth(1000);
    if (existing) {
      this.attached = true;
      this._appendLog(`[backend] attaching to existing backend on port ${this.port}`);
      this._emit("starting", "Found a backend already running.");
      this._pollUntilReady();
      return;
    }

    const { backendDir, venvPython } = this._resolvePaths();
    if (!fs.existsSync(venvPython)) {
      this._emit(
        "failed",
        "Setup isn't finished — the backend environment is missing. Please run the installer (scripts/install) once.",
      );
      return;
    }

    this._emit("starting", "Starting MeetingBro…");
    this._appendLog(`[backend] spawning ${venvPython} -m meetingbro.main (port ${this.port})`);

    try {
      this.child = spawn(venvPython, ["-m", "meetingbro.main"], {
        cwd: backendDir,
        env: { ...process.env, ...env, MEETINGBRO_PORT: String(this.port) },
        windowsHide: true,
      });
    } catch (err) {
      this._emit("failed", `Could not start the backend: ${err.message}`);
      return;
    }

    this.child.stdout.on("data", (d) => this._appendLog(d.toString().trimEnd()));
    this.child.stderr.on("data", (d) => this._appendLog(d.toString().trimEnd()));
    this.child.on("exit", (code, signal) => {
      this.childExited = true;
      this._appendLog(`[backend] process exited code=${code} signal=${signal}`);
      if (!this.quitting && this.state !== "ready") {
        this._emit("failed", "The backend stopped unexpectedly. Check the log for details.");
      } else if (!this.quitting && this.state === "ready") {
        this._emit("failed", "The backend stopped. Click Retry to restart it.");
      }
    });

    this._pollUntilReady();
  }

  // Poll /health until the model is ready (or the process dies). We key off
  // process liveness + the `ready` flag, never a fixed timeout, because the
  // first run downloads the speech model and can legitimately take minutes.
  async _pollUntilReady() {
    const token = ++this._pollToken;
    while (!this.quitting && token === this._pollToken) {
      if (this.child && this.childExited) return; // exit handler already emitted failed
      const health = await this._probeHealth(1500);
      if (token !== this._pollToken) return;
      if (health) {
        if (health.phase === "error") {
          this._emit("failed", health.detail || "The speech model failed to load.");
          return;
        }
        if (health.ready) {
          this._emit("ready");
          return;
        }
        // alive but model still loading/downloading
        this._emit("downloading-model", "Preparing the speech model (first run may download ~460 MB)…");
      }
      await sleep(HEALTH_POLL_INTERVAL_MS);
    }
  }

  _killTree() {
    if (!this.child || this.childExited) return;
    const pid = this.child.pid;
    if (process.platform === "win32") {
      try {
        spawn("taskkill", ["/F", "/T", "/PID", String(pid)], { windowsHide: true });
      } catch {
        try {
          this.child.kill();
        } catch {
          /* ignore */
        }
      }
    } else {
      try {
        this.child.kill("SIGTERM");
        setTimeout(() => {
          if (!this.childExited) {
            try {
              this.child.kill("SIGKILL");
            } catch {
              /* ignore */
            }
          }
        }, 4000);
      } catch {
        /* ignore */
      }
    }
  }

  // Stop the spawned backend (no-op if we attached to a foreign one).
  async stop() {
    this.quitting = true;
    this._pollToken++; // cancel any in-flight poll
    if (this.attached) {
      this._emit("stopped");
      return;
    }
    this._killTree();
    // Give the tree a moment to exit so a follow-up restart binds the port cleanly.
    for (let i = 0; i < 20 && this.child && !this.childExited; i += 1) {
      await sleep(100);
    }
    this._emit("stopped");
  }

  // Restart with fresh env (used after the user changes startup-only settings).
  async restart({ env = {}, port } = {}) {
    await this.stop();
    await this.start({ env, port: port ?? this.port });
  }
}

module.exports = { BackendSupervisor };
