/*
 * Renderer-side terminal driver.
 *
 * Mounts an xterm.js terminal inside #terminal, attaches a FitAddon so the
 * grid auto-fills the panel, and wires it to the PTY in the main process
 * through window.basherTerminalApi (exposed via preload.js).
 *
 * Exports a small module on window.basherTerminal so renderer.js can:
 *   - send agent-suggested commands into the live terminal (writeCommand)
 *   - run one-shot commands and read back stdout/stderr (exec)
 */
(function () {
  const Terminal = (window.Terminal || (window.xterm && window.xterm.Terminal));
  const FitAddonNS =
    window.FitAddon ||
    (window.addonFit && window.addonFit.FitAddon) ||
    null;
  const FitAddon =
    typeof FitAddonNS === "function" ? FitAddonNS : (FitAddonNS && FitAddonNS.FitAddon);

  const hostEl = document.getElementById("terminal");
  const infoEl = document.getElementById("termInfo");
  const newBtn = document.getElementById("termNewBtn");
  const clearBtn = document.getElementById("termClearBtn");
  const killBtn = document.getElementById("termKillBtn");

  if (!Terminal || !hostEl) {
    console.warn("xterm.js Terminal not available; terminal panel disabled");
    if (infoEl) infoEl.textContent = "xterm.js not loaded";
    return;
  }

  const term = new Terminal({
    cursorBlink: true,
    fontFamily:
      'ui-monospace, "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace',
    fontSize: 13,
    theme: {
      background: "#0b0e14",
      foreground: "#e8edf2",
      cursor: "#7aa2f7",
      selectionBackground: "#264f78",
    },
    convertEol: false,
    scrollback: 5000,
    allowProposedApi: true,
  });

  const fit = FitAddon ? new FitAddon() : null;
  if (fit) term.loadAddon(fit);
  term.open(hostEl);
  try { if (fit) fit.fit(); } catch (_) { /* layout not ready yet */ }

  let sessionId = null;
  let unbindData = null;
  let unbindExit = null;

  function setInfo(text) {
    if (infoEl) infoEl.textContent = text || "";
  }

  function teardown() {
    if (typeof unbindData === "function") {
      try { unbindData(); } catch (_) { /* noop */ }
      unbindData = null;
    }
    if (typeof unbindExit === "function") {
      try { unbindExit(); } catch (_) { /* noop */ }
      unbindExit = null;
    }
  }

  async function createSession() {
    teardown();
    term.reset();
    setInfo("starting shell...");

    const cols = term.cols;
    const rows = term.rows;
    const info = await window.basherTerminalApi.create({ cols, rows });
    sessionId = info.id;
    setInfo(`pid ${info.pid} • ${info.shell}`);

    unbindData = window.basherTerminalApi.onData(({ id, data }) => {
      if (id === sessionId) term.write(data);
    });

    unbindExit = window.basherTerminalApi.onExit(({ id, exitCode, signal }) => {
      if (id !== sessionId) return;
      term.writeln(
        `\r\n\x1b[33m[shell exited code=${exitCode} signal=${signal || "none"}]\x1b[0m`
      );
      setInfo(`exited (code ${exitCode})`);
      sessionId = null;
    });

    term.onData((data) => {
      if (sessionId) window.basherTerminalApi.write(sessionId, data);
    });

    term.onResize(({ cols, rows }) => {
      if (sessionId) window.basherTerminalApi.resize(sessionId, cols, rows);
    });
  }

  function refit() {
    try { if (fit) fit.fit(); } catch (_) { /* noop */ }
  }

  window.addEventListener("resize", refit);
  // ResizeObserver catches panel size changes when the grid reflows.
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(refit);
    ro.observe(hostEl);
  }

  newBtn?.addEventListener("click", () => { createSession().catch(reportError); });
  clearBtn?.addEventListener("click", () => { term.clear(); });
  killBtn?.addEventListener("click", async () => {
    if (!sessionId) return;
    try {
      await window.basherTerminalApi.dispose(sessionId);
    } catch (e) {
      reportError(e);
    }
  });

  function reportError(err) {
    const msg = err && err.message ? err.message : String(err);
    term.writeln(`\r\n\x1b[31m[terminal error] ${msg}\x1b[0m`);
    setInfo("error");
  }

  // Public API for renderer.js (agent <-> terminal bridge)
  window.basherTerminal = {
    isReady: () => Boolean(sessionId),
    focus: () => term.focus(),
    async ensureSession() {
      if (!sessionId) await createSession();
      return sessionId;
    },
    async writeCommand(command, { run = true } = {}) {
      await this.ensureSession();
      const line = String(command || "");
      window.basherTerminalApi.write(sessionId, line + (run ? "\r" : ""));
      term.focus();
    },
    async exec(command, opts) {
      return window.basherTerminalApi.exec(command, opts);
    },
  };

  // Auto-start a default shell on load.
  createSession().catch(reportError);
})();
