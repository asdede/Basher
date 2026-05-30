const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const terminalPty = require("./terminal-pty");

const API_BASE = process.env.BASHER_API_URL || "http://localhost:8000";

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

ipcMain.handle("agent:run", async (_event, payload) => {
  const res = await fetch(`${API_BASE}/agent/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || `Request failed (${res.status})`);
  }
  return JSON.parse(text);
});

ipcMain.handle("agent:session", async (_event, sessionId) => {
  const id = String(sessionId || "").trim();
  if (!id) {
    throw new Error("session_id required");
  }
  const res = await fetch(`${API_BASE}/agent/session/${encodeURIComponent(id)}`);
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || `Request failed (${res.status})`);
  }
  return JSON.parse(text);
});

ipcMain.handle("agent:toolResults", async (_event, payload) => {
  const res = await fetch(`${API_BASE}/agent/tool-results`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || `Request failed (${res.status})`);
  }
  return JSON.parse(text);
});

terminalPty.registerIpc(ipcMain);

app.whenReady().then(() => {
  createWindow();

  app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", function () {
  terminalPty.disposeAll();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  terminalPty.disposeAll();
});
