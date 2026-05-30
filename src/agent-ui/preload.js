const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("basherAgentApi", {
  runAgent: (payload) => ipcRenderer.invoke("agent:run", payload),
  getSession: (sessionId) => ipcRenderer.invoke("agent:session", sessionId),
  submitToolResults: (payload) => ipcRenderer.invoke("agent:toolResults", payload),
});

contextBridge.exposeInMainWorld("basherTerminalApi", {
  create: (opts) => ipcRenderer.invoke("terminal:create", opts || {}),
  write: (id, data) => ipcRenderer.invoke("terminal:input", { id, data }),
  resize: (id, cols, rows) =>
    ipcRenderer.invoke("terminal:resize", { id, cols, rows }),
  dispose: (id) => ipcRenderer.invoke("terminal:dispose", { id }),
  exec: (command, opts) =>
    ipcRenderer.invoke("terminal:exec", { command, ...(opts || {}) }),
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
