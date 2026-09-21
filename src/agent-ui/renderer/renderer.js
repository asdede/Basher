/*
 * basher — dual-mode terminal panel.
 *
 *   AGENT mode  (alt-screen)  — talks to /agent/run, REPL with summary +
 *                               next_step caching.
 *   SHELL mode  (main screen) — a real bash via node-pty.
 *
 * Toggle with F2. We ride on the same trick `vim`/`htop` use: enter the
 * alternate screen for the agent UI, leave it to reveal the live bash
 * session. Each mode keeps its own scrollback; they never bleed into each
 * other.
 *
 * Why a fresh agent session per turn (and a cached `summary` instead)?
 *   The driving model is a local tool-capable Qwen (default qwen3.5:4b).
 *   Long transcripts still hurt more than they help — relevance dilutes,
 *   schema drift creeps in, and tool calls start hallucinating flags.
 *   A short compressed memory keeps continuity without the long-context tax.
 *
 *   `next_step` is *only* used when the user presses Enter on an empty
 *   agent prompt ("return prompt"). Any non-empty input wins.
 */

(function () {
  // ----- xterm setup ------------------------------------------------------
  const Terminal = window.Terminal || (window.xterm && window.xterm.Terminal);
  const FitAddonNS =
    window.FitAddon || (window.addonFit && window.addonFit.FitAddon) || null;
  const FitAddon =
    typeof FitAddonNS === "function"
      ? FitAddonNS
      : FitAddonNS && FitAddonNS.FitAddon;

  const hostEl = document.getElementById("terminal");
  if (!Terminal || !hostEl) {
    document.body.textContent = "xterm.js failed to load.";
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
      black: "#1a1b26",
      red: "#f7768e",
      green: "#9ece6a",
      yellow: "#e0af68",
      blue: "#7aa2f7",
      magenta: "#bb9af7",
      cyan: "#7dcfff",
      white: "#a9b1d6",
      brightBlack: "#414868",
      brightRed: "#f7768e",
      brightGreen: "#9ece6a",
      brightYellow: "#e0af68",
      brightBlue: "#7aa2f7",
      brightMagenta: "#bb9af7",
      brightCyan: "#7dcfff",
      brightWhite: "#c0caf5",
    },
    convertEol: true,
    scrollback: 8000,
    allowProposedApi: true,
  });

  const fit = FitAddon ? new FitAddon() : null;
  if (fit) term.loadAddon(fit);
  term.open(hostEl);
  try {
    if (fit) fit.fit();
  } catch (_) {
    /* layout not ready */
  }

  // ----- ANSI helpers -----------------------------------------------------
  const ESC = "\x1b[";
  const RESET = `${ESC}0m`;
  const ALT_SCREEN_ON = "\x1b[?1049h";
  const ALT_SCREEN_OFF = "\x1b[?1049l";
  const c = {
    dim: (s) => `${ESC}2m${s}${RESET}`,
    bold: (s) => `${ESC}1m${s}${RESET}`,
    red: (s) => `${ESC}31m${s}${RESET}`,
    green: (s) => `${ESC}32m${s}${RESET}`,
    yellow: (s) => `${ESC}33m${s}${RESET}`,
    blue: (s) => `${ESC}34m${s}${RESET}`,
    magenta: (s) => `${ESC}35m${s}${RESET}`,
    cyan: (s) => `${ESC}36m${s}${RESET}`,
  };

  function write(s) {
    term.write(s);
  }
  function writeln(s) {
    term.writeln(s);
  }

  // ----- Mode state -------------------------------------------------------
  // The mode machine lives at the very top of input handling. Both modes
  // are "live" simultaneously underneath; we just gate IO routing.
  const MODE = { AGENT: "agent", SHELL: "shell" };
  let mode = MODE.AGENT;

  // PTY plumbing — bash runs the whole time even while the user is in
  // agent mode, so commands they kicked off keep making progress.
  let ptyId = null;
  let ptyAlive = false;
  // Bounded buffer for pty output that arrives while we're in agent mode.
  // 256 KiB is plenty for "I came back from a 5-minute build" without
  // letting a runaway `yes` eat the renderer's memory.
  const PTY_BUF_LIMIT = 256 * 1024;
  let ptyBuffer = "";

  function setWindowTitle() {
    const t = mode === MODE.AGENT ? "basher — agent (F2: shell)" : "basher — shell (F2: agent)";
    try {
      document.title = t;
    } catch (_) {
      /* noop */
    }
  }

  // ----- Banner -----------------------------------------------------------
  // The model line is rendered after a one-shot /health probe so the
  // banner can't lie about which model is actually serving requests.
  // (Old banner hard-coded "qwen2.5-coder:1.5b" which was wrong even
  // when the API was wired to the 7b — confusing to debug.)
  function banner() {
    writeln("");
    writeln(
      c.bold(c.cyan(" basher ")) +
        c.dim(" — Fedora bash agent ") +
        c.dim("(checking model…)")
    );
    writeln(
      c.dim("   ") +
        c.bold("F2") +
        c.dim(" toggles between agent REPL and a real bash shell.")
    );
    writeln(
      c.dim(
        "   in agent mode: type a question and press Enter; empty Enter " +
          "re-runs the cached next-step."
      )
    );
    writeln(
      c.dim("   ctrl-c cancels the current line / agent turn, ctrl-l clears.")
    );
    writeln("");
  }

  async function probeAndUpdateBannerModel() {
    let label = "model unknown";
    try {
      const h = await window.basherAgentApi.health();
      const model = h && h.model ? String(h.model) : "";
      const ctx = h && h.num_ctx ? `, ctx=${h.num_ctx}` : "";
      if (model) label = `${model}${ctx}`;
    } catch (_) {
      label = "API unreachable";
    }
    // The banner line we want to update is 5 lines above the current
    // prompt (banner is 5 writeln calls; cursor is on the prompt line).
    // We can't really edit a single line in xterm without ANSI cursor
    // moves; do it cleanly when idle and the buffer is empty.
    if (mode === MODE.AGENT && state === STATE.IDLE && buf === "") {
      // Save cursor, move 5 lines up to the banner's model line, clear
      // it, repaint with the real model label, then restore the cursor.
      write("\x1b7"); // save cursor
      write("\r" + `${ESC}5A` + `${ESC}2K`);
      write(
        c.bold(c.cyan(" basher ")) +
          c.dim(" — Fedora bash agent (") +
          c.cyan(label) +
          c.dim(")")
      );
      write("\x1b8"); // restore cursor
    }
  }

  // ----- Agent REPL state -------------------------------------------------
  const STATE = {
    IDLE: "idle",
    RUNNING: "running",
    DECISION: "decision",
  };
  let state = STATE.IDLE;
  let pendingDecisionResolver = null;
  let pendingDecisionAllowed = null;

  let buf = "";
  let cursor = 0;
  const promptText = c.green("basher$ ");

  const history = [];
  const HISTORY_LIMIT = 200;
  let historyIdx = -1;
  let savedBuf = "";

  let cachedSummary = "";
  let cachedNextStep = "";
  // Rolling last-N verbatim memory. The summary alone loses too much
  // ("yes, do that", "now from that result...") to be enough on its own.
  // 3 is a sweet spot for the small model: enough continuity, not so
  // much that the context starts crowding out the actual question.
  const PRIOR_TURNS_KEEP = 3;
  const recentTurns = []; // [{ user, answer }]

  // Consecutive schema-failure counter. When the model fails several
  // turns in a row we assume the rolling memory is poisoning new turns
  // ("find_folder" style placeholder after an exit=1 search) and wipe
  // the soft memory caches. Hard transcripts inside in-flight sessions
  // are untouched — we only flush the cross-session memory the renderer
  // owns. Threshold of 2 because a single fluke shouldn't nuke memory
  // the user has been building up across several good turns.
  let consecutiveFailures = 0;
  const FAILURE_FLUSH_THRESHOLD = 2;

  function showPrompt({ withHint = true } = {}) {
    if (withHint && cachedNextStep) {
      writeln(
        c.dim("  ↳ next: ") +
          c.cyan(cachedNextStep) +
          c.dim("   (press Enter to run, or type a new question)")
      );
    }
    write(promptText);
    buf = "";
    cursor = 0;
    historyIdx = -1;
    savedBuf = "";
    state = STATE.IDLE;
  }

  function redrawLine() {
    write("\r" + `${ESC}2K`);
    write(promptText + buf);
    const trailing = buf.length - cursor;
    if (trailing > 0) write(`${ESC}${trailing}D`);
  }

  // ----- Thinking trace ---------------------------------------------------
  // The server enforces "think at least twice before answering": when the
  // model would commit a final_answer straight off a single pass, it
  // forces a reflection round and returns BOTH thoughts as
  // `thinking_steps`. Tool-mediated turns naturally span >= 2 passes
  // (initial plan + post-tool reasoning) and arrive with `thinking_steps`
  // empty — for those we just show `thoughts` as before.
  function renderThinkingPasses(turn) {
    if (!turn) return;
    const stepsRaw = Array.isArray(turn.thinking_steps) ? turn.thinking_steps : [];
    const steps = stepsRaw
      .map((t) => String(t || "").trim())
      .filter((t) => t.length > 0);
    if (steps.length === 0) {
      const single = String(turn.thoughts || "").trim();
      if (single) writeln(c.magenta("• thinking ") + c.dim(single));
      return;
    }
    if (steps.length === 1) {
      writeln(c.magenta("• thinking ") + c.dim(steps[0]));
      return;
    }
    steps.forEach((thought, idx) => {
      writeln(
        c.magenta(`• thinking [pass ${idx + 1}/${steps.length}] `) +
          c.dim(thought)
      );
    });
  }

  // ----- Markdown-ish rendering for the final answer ----------------------
  function renderFinalAnswer(md) {
    const src = String(md || "");
    if (!src) {
      writeln(c.dim("(no answer)"));
      return;
    }
    const lines = src.split(/\r?\n/);
    let inFence = false;
    for (const raw of lines) {
      if (/^\s*```/.test(raw)) {
        inFence = !inFence;
        writeln(c.dim(raw));
        continue;
      }
      if (inFence) {
        writeln(c.cyan(raw));
        continue;
      }
      if (/^\s*#+\s+/.test(raw)) {
        writeln(c.bold(c.blue(raw)));
        continue;
      }
      writeln(raw.replace(/`([^`]+)`/g, (_m, code) => c.cyan(code)));
    }
  }

  // ----- Tool approval ----------------------------------------------------
  function dangerColor(d) {
    if (d === "destructive") return c.red(d);
    if (d === "caution") return c.yellow(d);
    return c.green(d || "safe");
  }

  function printToolRequest(call) {
    writeln("");
    writeln(
      c.yellow("• tool ") +
        c.dim("(" + dangerColor(call.danger) + ")") +
        (call.purpose ? " " + c.dim(call.purpose) : "")
    );
    writeln("    " + c.bold("$ ") + call.command);
    if (call.banned_reason) {
      writeln("    " + c.red("⛔ banned: ") + call.banned_reason);
    }
  }

  function printToolResult(call, result) {
    const ok = !result.timedOut && result.code === 0;
    writeln(
      "    " +
        (ok ? c.green("✓") : c.red("✗")) +
        " exit=" +
        String(result.code) +
        (result.timedOut ? c.red(" timed-out") : "") +
        (result.signal ? c.dim(" signal=" + result.signal) : "")
    );
    const out = (result.stdout || "").trimEnd();
    const err = (result.stderr || "").trimEnd();
    if (out) for (const line of out.split(/\r?\n/)) writeln("    " + line);
    if (err) for (const line of err.split(/\r?\n/)) writeln("    " + c.red(line));
    if (!out && !err) writeln("    " + c.dim("(no output)"));
  }

  async function approveCall(call) {
    if (call.banned_reason) {
      writeln("    " + c.dim("auto-skipped (banned)"));
      return "skip";
    }
    return new Promise((resolve) => {
      const allowed = new Set(["r", "s", "e"]);
      pendingDecisionAllowed = allowed;
      pendingDecisionResolver = (key) => {
        pendingDecisionAllowed = null;
        pendingDecisionResolver = null;
        if (key === "s") return resolve("skip");
        if (key === "e") {
          state = STATE.IDLE;
          editToolCommand(call.command).then((next) => {
            if (next == null) return resolve("skip");
            resolve({ decision: "run-edited", command: next });
          });
          return;
        }
        resolve("run");
      };
      writeln(
        "    " +
          c.dim("approve? ") +
          c.bold("[r]") +
          c.dim("un  ") +
          c.bold("[s]") +
          c.dim("kip  ") +
          c.bold("[e]") +
          c.dim("dit")
      );
      write("    " + c.dim("> "));
      state = STATE.DECISION;
    });
  }

  let editResolver = null;
  function editToolCommand(initial) {
    return new Promise((resolve) => {
      writeln("");
      write(c.dim("    edit> "));
      buf = String(initial || "");
      cursor = buf.length;
      write(buf);
      editResolver = (final) => {
        editResolver = null;
        resolve(final);
      };
    });
  }

  // ----- Agent driver -----------------------------------------------------
  let abortRequested = false;
  let activeRun = null;

  async function executeToolCall(call) {
    writeln(c.dim("    running..."));
    return window.basherShell.exec(call.command, { timeoutMs: 60_000 });
  }

  async function runAgent(query) {
    activeRun = { sessionId: null, query };
    abortRequested = false;
    state = STATE.RUNNING;

    let turn;
    try {
      const payload = {
        query,
        mode: "interactive",
        debug: false,
        base_model: false,
        session_id: null,
        // null → server auto-routes RAG (docs for how-to; tools-only for live state)
        use_retrieval: null,
        prior_summary: cachedSummary || null,
        // Verbatim last-N memory. Sent fresh on every prompt so the
        // server stays stateless w.r.t. our chat history.
        prior_turns: recentTurns.length ? recentTurns.slice() : null,
      };
      turn = await window.basherAgentApi.runAgent(payload);
    } catch (err) {
      writeln(c.red("error: ") + (err && err.message ? err.message : String(err)));
      state = STATE.IDLE;
      return;
    }
    activeRun.sessionId = turn.session_id;

    while (turn && !turn.done) {
      if (abortRequested) {
        writeln(c.dim("(aborted)"));
        break;
      }
      renderThinkingPasses(turn);
      const results = [];
      for (const call of turn.tool_calls || []) {
        if (abortRequested) break;
        printToolRequest(call);
        const decision = await approveCall(call);
        if (decision === "skip") {
          writeln("    " + c.dim("skipped"));
          results.push({
            tool_call_id: call.id,
            command: call.command,
            stdout: "",
            stderr: "[skipped by user]",
            exit_code: -2,
            timed_out: false,
          });
          continue;
        }
        let cmdToRun = call.command;
        if (decision && decision.decision === "run-edited") {
          cmdToRun = decision.command;
          writeln("    " + c.dim("edited: ") + cmdToRun);
        }
        let exec;
        try {
          exec = await executeToolCall({ ...call, command: cmdToRun });
        } catch (err) {
          const msg = err && err.message ? err.message : String(err);
          writeln("    " + c.red("[client error] ") + msg);
          results.push({
            tool_call_id: call.id,
            command: cmdToRun,
            stdout: "",
            stderr: `[client error] ${msg}`,
            exit_code: -1,
            timed_out: false,
          });
          continue;
        }
        printToolResult({ ...call, command: cmdToRun }, exec);
        results.push({
          tool_call_id: call.id,
          command: cmdToRun,
          stdout: exec.stdout || "",
          stderr: exec.stderr || "",
          exit_code: typeof exec.code === "number" ? exec.code : -1,
          timed_out: Boolean(exec.timedOut),
        });
      }
      if (!results.length) {
        writeln(c.red("no tool results produced; halting loop"));
        break;
      }
      try {
        turn = await window.basherAgentApi.submitToolResults({
          session_id: turn.session_id,
          results,
          debug: false,
        });
      } catch (err) {
        writeln(c.red("error: ") + (err && err.message ? err.message : String(err)));
        break;
      }
    }

    if (turn && turn.done) {
      renderThinkingPasses(turn);
      if (turn.retry_reason) {
        // The server had to send a corrective nudge to the model. Show
        // it so the user knows why latency might have been a hair
        // higher — and so prompt-engineers can see this in the wild.
        const labels = {
          thinking_only: "model planned but didn't act — nudged it to run the command",
          refusal: "model refused — nudged it to drop the safety reflex",
          echo: "model echoed the question — nudged it for new info",
          field_leak: "model returned a schema keyword — nudged it for a real answer",
          no_tool_for_state: "model guessed instead of running a tool — nudged it to actually search",
        };
        writeln(
          c.dim("  ↳ " + (labels[turn.retry_reason] || "model needed a nudge"))
        );
      }
      writeln("");
      const ok = isUsefulAnswer(turn);
      if (ok) {
        renderFinalAnswer(turn.final_answer);
        // A real answer resets the failure streak — the soft memory is
        // clearly still useful.
        consecutiveFailures = 0;
        // Push verbatim memory: keep the last N and feed them on the
        // next /agent/run call.
        recentTurns.push({
          user: query,
          answer: String(turn.final_answer || ""),
        });
        while (recentTurns.length > PRIOR_TURNS_KEEP) recentTurns.shift();
        // Only feed the summarizer real exchanges — junk in, junk out
        // and the cached next_step starts hallucinating ("show podman
        // logs" pattern-matching off the system prompt's few-shots).
        summarizeInBackground(query, turn.final_answer);
      } else {
        // Drop any stale cache so empty Enter on the next prompt does
        // not keep replaying a broken turn.
        cachedNextStep = "";
        // Server now substitutes raw placeholder tokens (like
        // "find_folder") with a clean English explanation before this
        // payload arrives, so `turn.final_answer` is either empty, a
        // short legacy synthetic, or a human-readable explanation. We
        // render the explanation through `renderFinalAnswer` (markdown
        // pass) and prefix it with a compact "(model failed: <reason>)"
        // tag so the user immediately knows the answer is the agent
        // explaining its own failure, not a real result.
        const said = String(turn.final_answer || "").trim();
        const reasonLabels = {
          field_leak: "field_leak (placeholder token)",
          echo: "echo (paraphrased question)",
          refusal: "refusal",
          thinking_only: "thinking_only (planned without acting)",
          empty: "empty response",
          non_json: "malformed JSON",
          no_tool_for_state: "no_tool_for_state (guessed instead of running a tool)",
        };
        const reasonText = reasonLabels[turn.failure_reason] ||
          turn.failure_reason || "unusable response";
        writeln(c.yellow("• model failed ") + c.dim("(" + reasonText + ")"));
        if (said) {
          renderFinalAnswer(said);
        } else {
          writeln(c.dim("(no text returned)"));
        }
        writeln(
          c.dim("→ try rephrasing, or press F2 to run a command directly.")
        );
        // Wipe the soft memory caches if failures keep stacking up. The
        // most common cause of stuck failures is the previous turn's
        // recap or recentTurns biasing the model into the same wrong
        // pattern (the "find_folder after exit=1" loop the user hit).
        consecutiveFailures += 1;
        if (consecutiveFailures >= FAILURE_FLUSH_THRESHOLD) {
          if (cachedSummary || recentTurns.length) {
            writeln(
              c.dim(
                "  ↳ cleared rolling memory after " +
                  consecutiveFailures +
                  " consecutive failures (the cache may have been poisoning new turns)."
              )
            );
          }
          cachedSummary = "";
          cachedNextStep = "";
          recentTurns.length = 0;
          consecutiveFailures = 0;
        }
      }
    }

    state = STATE.IDLE;
    writeln("");
    showPrompt({ withHint: false });
  }

  // A "useful" turn has schema_ok=true AND a non-empty, non-placeholder
  // final_answer. The placeholder check covers the "(no answer)" /
  // "(no response)" sentinels older API revisions emitted, plus our own
  // synthetic strings. We never summarize/cache a non-useful turn.
  function isUsefulAnswer(turn) {
    if (!turn) return false;
    if (turn.schema_ok === false) return false;
    const a = String(turn.final_answer || "").trim();
    if (!a) return false;
    if (a.startsWith("(") && a.endsWith(")") && a.length < 60) return false;
    return true;
  }

  async function summarizeInBackground(query, finalAnswer) {
    if (!finalAnswer) return;
    try {
      const out = await window.basherAgentApi.summarize({
        query,
        final_answer: finalAnswer,
        prior_summary: cachedSummary || null,
      });
      const summary = String(out && out.summary ? out.summary : "").trim();
      const nextStep = String(out && out.next_step ? out.next_step : "").trim();
      if (summary) cachedSummary = summary;
      if (nextStep) {
        cachedNextStep = nextStep;
        // Only update the visible hint if we're in agent mode, idle, with
        // an empty buffer. Otherwise the user is mid-typing and we'd
        // clobber their input.
        if (mode === MODE.AGENT && state === STATE.IDLE && buf === "") {
          write("\r" + `${ESC}2K`);
          write(`${ESC}1A` + `${ESC}2K`);
          writeln(
            c.dim("  ↳ next: ") +
              c.cyan(cachedNextStep) +
              c.dim("   (press Enter to run, or type a new question)")
          );
          write(promptText);
        }
      }
    } catch (_) {
      /* best-effort */
    }
  }

  // ----- Submit handling --------------------------------------------------
  function submitLine() {
    const line = buf;
    buf = "";
    cursor = 0;
    historyIdx = -1;
    savedBuf = "";
    writeln("");

    if (editResolver) {
      const r = editResolver;
      const trimmed = line.trim();
      r(trimmed || null);
      return;
    }

    const query = decideQuery(line);
    if (!query) {
      showPrompt({ withHint: false });
      return;
    }

    if (line.trim() === "" && query !== "") {
      writeln(c.dim("(running cached next-step) ") + c.cyan(query));
    }
    if (history[history.length - 1] !== query) {
      history.push(query);
      if (history.length > HISTORY_LIMIT) history.shift();
    }
    cachedNextStep = "";
    runAgent(query);
  }

  function decideQuery(line) {
    const trimmed = line.trim();
    if (trimmed) return trimmed;
    if (cachedNextStep) return cachedNextStep;
    return "";
  }

  // ----- PTY lifecycle ----------------------------------------------------
  // The pty session is shared by the whole window. It survives mode
  // toggles. If bash exits, we recreate so F2 always lands somewhere
  // useful.
  async function ensurePty() {
    if (ptyId && ptyAlive) return ptyId;
    const info = await window.basherShell.create({
      cols: term.cols,
      rows: term.rows,
    });
    ptyId = info.id;
    ptyAlive = true;
    return ptyId;
  }

  function ptyWrite(data) {
    if (!ptyId || !ptyAlive) return;
    window.basherShell.write(ptyId, data).catch(() => {});
  }

  function ptyResize(cols, rows) {
    if (!ptyId || !ptyAlive) return;
    window.basherShell.resize(ptyId, cols, rows).catch(() => {});
  }

  // ----- Mode toggle ------------------------------------------------------
  function enterAgentMode({ initial = false } = {}) {
    if (mode === MODE.AGENT && !initial) return;
    mode = MODE.AGENT;
    write(ALT_SCREEN_ON);
    setWindowTitle();
    if (initial) {
      banner();
      showPrompt({ withHint: false });
    }
    // Don't redraw on subsequent toggles — the alt-screen kept our state.
  }

  async function enterShellMode() {
    if (mode === MODE.SHELL) return;
    // If bash died while we were in agent mode, transparently respawn so
    // F2 always lands somewhere useful.
    if (!ptyAlive) {
      try {
        await ensurePty();
      } catch (_) {
        /* leave the user a dead shell, they'll see the message */
      }
    }
    mode = MODE.SHELL;
    write(ALT_SCREEN_OFF);
    setWindowTitle();
    if (ptyBuffer.length > 0) {
      term.write(ptyBuffer);
      ptyBuffer = "";
    }
    // Nudge bash to redraw its prompt (it was sitting idle, but we may
    // have lost the last prompt to the alt-screen scrollback).
    ptyWrite("");
  }

  function toggleMode() {
    if (mode === MODE.AGENT) enterShellMode();
    else enterAgentMode();
  }

  // ----- Key dispatch -----------------------------------------------------
  // F2 in either xterm flavor: \x1bOQ (xterm-application-keypad) or \x1b[12~.
  function isToggleKey(s) {
    return s === "\x1bOQ" || s === "\x1b[12~";
  }

  function handleData(data) {
    // F2 wins everywhere — including mid-agent-run, mid-decision-prompt,
    // mid-bash-pipe-yes. The toggle never blocks.
    if (isToggleKey(data)) {
      toggleMode();
      return;
    }
    // F2 might arrive at the front of a chunk along with other bytes
    // (rare, but possible with paste). Strip a leading toggle and handle
    // the rest.
    if (data.startsWith("\x1bOQ") || data.startsWith("\x1b[12~")) {
      toggleMode();
      data = data.slice(data.startsWith("\x1bOQ") ? 3 : 5);
      if (!data) return;
    }

    if (mode === MODE.SHELL) {
      ptyWrite(data);
      return;
    }
    handleAgentData(data);
  }

  function handleAgentData(data) {
    if (state === STATE.DECISION) {
      handleDecisionData(data);
      return;
    }
    if (state === STATE.RUNNING && !editResolver) {
      if (data === "\x03") {
        abortRequested = true;
        writeln("\n" + c.dim("(aborting...)"));
      }
      return;
    }

    let i = 0;
    while (i < data.length) {
      const ch = data[i];

      if (ch === "\x03") {
        if (editResolver) {
          writeln("");
          const r = editResolver;
          editResolver = null;
          r(null);
          return;
        }
        writeln("^C");
        showPrompt({ withHint: false });
        return;
      }

      if (ch === "\x0c") {
        term.clear();
        write(promptText + buf);
        const trailing = buf.length - cursor;
        if (trailing > 0) write(`${ESC}${trailing}D`);
        i++;
        continue;
      }

      if (ch === "\r" || ch === "\n") {
        submitLine();
        return;
      }

      if (ch === "\x7f" || ch === "\b") {
        if (cursor > 0) {
          buf = buf.slice(0, cursor - 1) + buf.slice(cursor);
          cursor--;
          redrawLine();
        }
        i++;
        continue;
      }

      if (ch === "\x1b" && data[i + 1] === "[") {
        const code = data[i + 2];
        if (code === "A") {
          if (history.length === 0) {
            i += 3;
            continue;
          }
          if (historyIdx === -1) savedBuf = buf;
          if (historyIdx === -1) historyIdx = history.length;
          if (historyIdx > 0) historyIdx--;
          buf = history[historyIdx];
          cursor = buf.length;
          redrawLine();
          i += 3;
          continue;
        }
        if (code === "B") {
          if (historyIdx === -1) {
            i += 3;
            continue;
          }
          historyIdx++;
          if (historyIdx >= history.length) {
            historyIdx = -1;
            buf = savedBuf;
          } else {
            buf = history[historyIdx];
          }
          cursor = buf.length;
          redrawLine();
          i += 3;
          continue;
        }
        if (code === "C") {
          if (cursor < buf.length) {
            cursor++;
            write(`${ESC}1C`);
          }
          i += 3;
          continue;
        }
        if (code === "D") {
          if (cursor > 0) {
            cursor--;
            write(`${ESC}1D`);
          }
          i += 3;
          continue;
        }
        // Skip unknown CSI sequences cleanly (\x1b[...letter or ~).
        let j = i + 2;
        while (j < data.length && !/[a-zA-Z~]/.test(data[j])) j++;
        i = j + 1;
        continue;
      }

      if (ch >= " " && ch !== "\x7f") {
        let j = i;
        while (j < data.length && data[j] >= " " && data[j] !== "\x7f") j++;
        const chunk = data.slice(i, j);
        buf = buf.slice(0, cursor) + chunk + buf.slice(cursor);
        cursor += chunk.length;
        redrawLine();
        i = j;
        continue;
      }

      i++;
    }
  }

  function handleDecisionData(data) {
    for (const ch of data) {
      const lower = ch.toLowerCase();
      if (ch === "\x03") {
        writeln("^C");
        const r = pendingDecisionResolver;
        pendingDecisionAllowed = null;
        pendingDecisionResolver = null;
        if (r) r("s");
        return;
      }
      if (pendingDecisionAllowed && pendingDecisionAllowed.has(lower)) {
        write(lower);
        writeln("");
        const r = pendingDecisionResolver;
        pendingDecisionAllowed = null;
        pendingDecisionResolver = null;
        if (r) r(lower);
        return;
      }
      write("\x07");
    }
  }

  term.onData(handleData);

  // ----- Resize: keep both surfaces aligned -------------------------------
  function refit() {
    try {
      if (fit) fit.fit();
    } catch (_) {
      /* noop */
    }
  }
  window.addEventListener("resize", refit);
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(refit);
    ro.observe(hostEl);
  }
  term.onResize(({ cols, rows }) => ptyResize(cols, rows));

  // ----- Boot -------------------------------------------------------------
  async function boot() {
    setWindowTitle();
    // Wire pty data first so we never miss output once bash starts up.
    window.basherShell.onData(({ id, data }) => {
      if (id !== ptyId) return;
      if (mode === MODE.SHELL) {
        term.write(data);
      } else if (ptyBuffer.length < PTY_BUF_LIMIT) {
        ptyBuffer += data;
        if (ptyBuffer.length > PTY_BUF_LIMIT) {
          // Cap with a visible marker so the user knows we dropped tail.
          ptyBuffer =
            ptyBuffer.slice(0, PTY_BUF_LIMIT) +
            "\r\n[basher: shell output truncated while in agent mode]\r\n";
        }
      }
    });
    window.basherShell.onExit(({ id, exitCode, signal }) => {
      if (id !== ptyId) return;
      ptyAlive = false;
      const note =
        `\r\n[shell exited code=${exitCode} signal=${signal || "none"}; ` +
        `press F2 twice to start a new one]\r\n`;
      if (mode === MODE.SHELL) {
        term.write(note);
      } else {
        ptyBuffer += note;
      }
    });

    // Spawn bash up-front so the user always has a live shell waiting on
    // F2, and so anything they leave running keeps making progress while
    // they're in agent mode.
    try {
      await ensurePty();
    } catch (err) {
      // Non-fatal — the agent panel works even without a shell. We just
      // can't toggle.
      // Show a single dim line in the agent banner area later.
      console.warn("pty spawn failed:", err);
    }

    enterAgentMode({ initial: true });
    term.focus();
    // Best-effort: replace the "(checking model…)" placeholder in the
    // banner with the actual model the API is configured for. Never
    // blocks boot — if the probe fails we leave the placeholder text.
    probeAndUpdateBannerModel().catch(() => {});
  }

  boot();
})();
