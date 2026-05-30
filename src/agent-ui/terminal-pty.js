/*
 * PTY manager for Basher agent-ui.
 *
 * - Spawns interactive shells via node-pty for the renderer's xterm panel.
 * - Bridges PTY <-> renderer via ipcMain on per-window channels.
 * - Provides an additional non-interactive `terminal:exec` channel so the
 *   agent can run a single command and capture its output (for info
 *   gathering: `uname -r`, `dnf list installed`, `systemctl status`, ...).
 *
 * Fedora notes:
 *   - default shell taken from $SHELL, falls back to /bin/bash.
 *   - $TERM is forced to xterm-256color so dnf/systemctl/ls output stays
 *     colored and ncurses tools (htop, vim, less) render correctly.
 */

const os = require("os");
const { spawn } = require("child_process");
const pty = require("node-pty");

const sessions = new Map(); // id -> { proc, wc }
let nextId = 1;

function defaultShell() {
  if (process.env.SHELL) return process.env.SHELL;
  if (process.platform === "win32") return "powershell.exe";
  return "/bin/bash";
}

function defaultEnv(extra) {
  const env = { ...process.env, ...(extra || {}) };
  env.TERM = env.TERM || "xterm-256color";
  env.COLORTERM = env.COLORTERM || "truecolor";
  env.LANG = env.LANG || "C.UTF-8";
  // Make the shell announce itself to children of the agent UI.
  env.BASHER_AGENT_UI = "1";
  return env;
}

function createSession(webContents, opts = {}) {
  const id = String(nextId++);
  const shell = opts.shell || defaultShell();
  const args = Array.isArray(opts.args) ? opts.args : ["-l"];
  const cwd = opts.cwd || os.homedir() || process.cwd();
  const cols = Number.isInteger(opts.cols) ? opts.cols : 100;
  const rows = Number.isInteger(opts.rows) ? opts.rows : 30;

  const proc = pty.spawn(shell, args, {
    name: "xterm-256color",
    cols,
    rows,
    cwd,
    env: defaultEnv(opts.env),
  });

  const session = { id, proc, wc: webContents };
  sessions.set(id, session);

  proc.onData((data) => {
    if (!webContents.isDestroyed()) {
      webContents.send("terminal:data", { id, data });
    }
  });

  proc.onExit(({ exitCode, signal }) => {
    if (!webContents.isDestroyed()) {
      webContents.send("terminal:exit", { id, exitCode, signal });
    }
    sessions.delete(id);
  });

  webContents.once("destroyed", () => {
    try {
      proc.kill();
    } catch (_) {
      // already gone
    }
    sessions.delete(id);
  });

  return { id, pid: proc.pid, shell, cwd };
}

function writeSession(id, data) {
  const s = sessions.get(String(id));
  if (!s) throw new Error(`unknown terminal id: ${id}`);
  s.proc.write(String(data ?? ""));
}

function resizeSession(id, cols, rows) {
  const s = sessions.get(String(id));
  if (!s) return;
  const c = Math.max(2, Number(cols) || 80);
  const r = Math.max(2, Number(rows) || 24);
  try {
    s.proc.resize(c, r);
  } catch (_) {
    // pty may already be exiting
  }
}

function killSession(id) {
  const s = sessions.get(String(id));
  if (!s) return false;
  try {
    s.proc.kill();
  } catch (_) {
    /* noop */
  }
  sessions.delete(String(id));
  return true;
}

/**
 * One-shot non-interactive command execution for the agent.
 * Uses child_process so we keep stdout/stderr cleanly separated.
 * Always runs through `bash -lc` so the agent gets the user's PATH and
 * profile-defined functions/aliases.
 */
function execOneShot(command, opts = {}) {
  return new Promise((resolve) => {
    const cmd = String(command || "").trim();
    if (!cmd) {
      resolve({ ok: false, code: -1, stdout: "", stderr: "empty command" });
      return;
    }
    const cwd = opts.cwd || os.homedir() || process.cwd();
    const timeoutMs = Math.max(1000, Number(opts.timeoutMs) || 30000);

    const child = spawn("/bin/bash", ["-lc", cmd], {
      cwd,
      env: defaultEnv(opts.env),
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const limit = 1024 * 1024; // 1 MiB cap per stream

    child.stdout.on("data", (b) => {
      if (stdout.length < limit) stdout += b.toString("utf8");
    });
    child.stderr.on("data", (b) => {
      if (stderr.length < limit) stderr += b.toString("utf8");
    });

    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGKILL");
      } catch (_) {
        /* noop */
      }
    }, timeoutMs);

    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        ok: false,
        code: -1,
        stdout,
        stderr: stderr + `\n[spawn error] ${err.message}`,
        timedOut,
      });
    });

    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        ok: !timedOut && code === 0,
        code: code ?? -1,
        signal: signal || null,
        stdout,
        stderr,
        timedOut,
      });
    });
  });
}

function registerIpc(ipcMain) {
  ipcMain.handle("terminal:create", (event, opts) =>
    createSession(event.sender, opts || {})
  );
  ipcMain.handle("terminal:input", (_event, payload) => {
    writeSession(payload?.id, payload?.data);
    return true;
  });
  ipcMain.handle("terminal:resize", (_event, payload) => {
    resizeSession(payload?.id, payload?.cols, payload?.rows);
    return true;
  });
  ipcMain.handle("terminal:dispose", (_event, payload) =>
    killSession(payload?.id)
  );
  ipcMain.handle("terminal:exec", (_event, payload) =>
    execOneShot(payload?.command, payload || {})
  );
}

function disposeAll() {
  for (const s of sessions.values()) {
    try {
      s.proc.kill();
    } catch (_) {
      /* noop */
    }
  }
  sessions.clear();
}

module.exports = { registerIpc, disposeAll };
