/*
 * Preload bridge for the basher terminal UI.
 *
 * The renderer is sandboxed (contextIsolation + no node integration); these
 * are the only escapes it has. We expose:
 *
 *   - basherAgentApi: HTTP calls to the FastAPI agent endpoints.
 *   - basherShell:    one-shot bash exec for tool calls. We deliberately do
 *                     not expose interactive PTY sessions — the renderer's
 *                     own xterm pane is the agent REPL, not a host shell.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("basherAgentApi", {
  health: () => ipcRenderer.invoke("agent:health"),
  runAgent: (payload) => ipcRenderer.invoke("agent:run", payload),
  submitToolResults: (payload) => ipcRenderer.invoke("agent:toolResults", payload),
  summarize: (payload) => ipcRenderer.invoke("agent:summarize", payload),
});

contextBridge.exposeInMainWorld("basherShell", {
  exec: (command, opts) =>
    ipcRenderer.invoke("terminal:exec", { command, ...(opts || {}) }),
  // Interactive PTY for the dual-mode panel (F2 toggles between agent
  // REPL and a real bash). The renderer drives an alt-screen agent UI on
  // top of a live bash session living in the main screen.
  create: (opts) => ipcRenderer.invoke("terminal:create", opts || {}),
  write: (id, data) => ipcRenderer.invoke("terminal:input", { id, data }),
  resize: (id, cols, rows) =>
    ipcRenderer.invoke("terminal:resize", { id, cols, rows }),
  dispose: (id) => ipcRenderer.invoke("terminal:dispose", { id }),
  onData: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("terminal:data", listener);
    return () => ipcRenderer.removeListener("terminal:data", listener);
  },
  onExit: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("terminal:exit", listener);
    return () => ipcRenderer.removeListener("terminal:exit", listener);
  },
});
