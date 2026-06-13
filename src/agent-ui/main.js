/*
 * Electron main process for the basher terminal UI.
 *
 * Responsibilities:
 *   - Open a single window that loads the xterm-only renderer.
 *   - Bridge the renderer to the FastAPI agent endpoints over HTTP.
 *   - Expose a one-shot bash exec for tool calls (terminal-pty.js).
 *
 * We no longer manage interactive PTY shell sessions for the renderer:
 * the xterm pane in the renderer is the agent REPL itself, driven by JS,
 * so no host shell is spawned underneath it.
 */

const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const dns = require("dns");
const terminalPty = require("./terminal-pty");

// Node 18+ defaults DNS resolution to "verbatim" order (whatever
// getaddrinfo returns first), and on glibc that means IPv6 first. The
// API container only listens on IPv4 (Docker's 0.0.0.0 mapping), so
// `localhost` -> `::1` opens a connection that the kernel immediately
// closes -> "fetch failed: SocketError: other side closed". Force IPv4
// preference for any URL we resolve.
try {
  dns.setDefaultResultOrder("ipv4first");
} catch (_) {
  /* older nodes don't have it; default to 127.0.0.1 below */
}

// Default to 127.0.0.1 (not localhost) as a belt-and-suspenders fix for
// the IPv6 issue in case BASHER_API_URL was never set explicitly.
const API_BASE = process.env.BASHER_API_URL || "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 5 * 60_000;

function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 720,
    backgroundColor: "#0b0e14",
    title: "basher",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.removeMenu();
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

async function postJson(pathname, payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}${pathname}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      // FastAPI HTTPException puts the human-readable message in
      // {"detail": "..."} — surface that, not the raw JSON envelope.
      let detail = text;
      try {
        const j = JSON.parse(text);
        if (j && typeof j.detail === "string") detail = j.detail;
        else if (j && j.detail) detail = JSON.stringify(j.detail);
      } catch (_) {
        /* not JSON, fall through */
      }
      throw new Error(detail || `Request failed (${res.status})`);
    }
    try {
      return JSON.parse(text);
    } catch (_) {
      throw new Error(`Invalid JSON from ${pathname}: ${text.slice(0, 200)}`);
    }
  } finally {
    clearTimeout(timer);
  }
}

ipcMain.handle("agent:run", (_e, payload) => postJson("/agent/run", payload));
ipcMain.handle("agent:toolResults", (_e, payload) =>
  postJson("/agent/tool-results", payload)
);
ipcMain.handle("agent:summarize", (_e, payload) =>
  postJson("/agent/summarize", payload)
);

terminalPty.registerIpc(ipcMain);

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  terminalPty.disposeAll();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  terminalPty.disposeAll();
});
