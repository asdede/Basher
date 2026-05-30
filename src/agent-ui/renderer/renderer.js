/*
 * Renderer driver for the Basher agent loop.
 *
 * Protocol (see src/api/agent.py):
 *   POST /agent/run -> { session_id, tool_calls, final_answer, done, ... }
 *   For each tool_call: execute via PTY exec, then POST results to
 *   /agent/tool-results. Repeat until done.
 *
 * Modes:
 *   interactive: tool calls require user approval (Run / Skip).
 *   autonomous: auto-run `safe` calls; `caution`/`destructive` still require
 *               explicit approval to avoid silent damage.
 */

function getEl(id) {
  return document.getElementById(id);
}

const modeEl = getEl("mode");
const debugEl = getEl("debug");
const baseModelEl = getEl("baseModel");
const sessionIdEl = getEl("sessionId");
const queryEl = getEl("query");
const statusEl = getEl("status");
const responseEl = getEl("response");
const stepsEl = getEl("steps");
const sessionEl = getEl("session");
const agentLogEl = getEl("agent-log");
const suggestedEl = getEl("suggested-commands");
const iterInfoEl = getEl("iterInfo");
const sendBtn = getEl("sendBtn");
const loadSessionBtn = getEl("loadSessionBtn");

const TOOL_TIMEOUT_MS = 60_000;

function setStatus(message) {
  statusEl.textContent = message;
}

function asText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Minimal markdown renderer: fenced code, inline code, bold. Safe enough
 * for assistant output without pulling a real library. */
function renderMarkdown(md) {
  const src = String(md || "");
  const fenced = src.replace(
    /```(\w+)?\n([\s\S]*?)```/g,
    (_m, lang, code) =>
      `<pre><code class="lang-${escapeHtml(lang || "")}">${escapeHtml(code)}</code></pre>`
  );
  return fenced
    .replace(/`([^`\n]+)`/g, (_m, c) => `<code>${escapeHtml(c)}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, (_m, c) => `<strong>${escapeHtml(c)}</strong>`)
    .replace(/\n/g, "<br/>");
}

function clearLog() {
  if (agentLogEl) agentLogEl.innerHTML = "";
}

function logEntry(kind, html) {
  if (!agentLogEl) return null;
  const div = document.createElement("div");
  div.className = `log-entry ${kind}`;
  div.innerHTML = html;
  agentLogEl.appendChild(div);
  agentLogEl.scrollTop = agentLogEl.scrollHeight;
  return div;
}

function appendThoughts(thoughts) {
  if (!thoughts) return;
  logEntry(
    "thoughts",
    `<span class="tag">thinking</span>${escapeHtml(thoughts)}`
  );
}

function appendSystem(message) {
  logEntry("system", `<span class="tag">info</span>${escapeHtml(message)}`);
}

function appendError(message) {
  logEntry("error", `<span class="tag">error</span>${escapeHtml(message)}`);
}

function appendToolRequest(call, controls) {
  const wrap = document.createElement("div");
  wrap.className = "log-entry tool";
  const dangerClass = `danger-${call.danger || "safe"}`;
  const banned = call.banned_reason ? ` ⛔ ${escapeHtml(call.banned_reason)}` : "";
  wrap.innerHTML =
    `<span class="tag">tool</span>` +
    `<span class="danger-tag ${dangerClass}">${escapeHtml(call.danger || "safe")}</span>` +
    (call.purpose ? `<span style="color:#98a4b3"> ${escapeHtml(call.purpose)}</span>` : "") +
    banned +
    `<div style="margin-top:6px"><code>$ ${escapeHtml(call.command)}</code></div>`;

  if (controls) {
    const actions = document.createElement("div");
    actions.className = "log-actions";
    for (const c of controls) {
      const b = document.createElement("button");
      b.textContent = c.label;
      b.className = c.className || "";
      b.addEventListener("click", () => c.onClick(b));
      actions.appendChild(b);
    }
    wrap.appendChild(actions);
  }

  agentLogEl.appendChild(wrap);
  agentLogEl.scrollTop = agentLogEl.scrollHeight;
  return wrap;
}

function appendToolResult(call, result) {
  const ok = !result.timedOut && result.code === 0;
  const summary =
    `<span class="tag">result</span>` +
    `<span style="color:${ok ? "#9ece6a" : "#f7768e"}">exit=${result.code}` +
    `${result.timedOut ? " (timed out)" : ""}${result.signal ? ` signal=${result.signal}` : ""}</span>` +
    `<div style="margin-top:4px;color:#98a4b3;font-size:11px">for <code>${escapeHtml(call.command)}</code></div>` +
    `<pre>${escapeHtml(
      [
        result.stdout ? `--- stdout ---\n${result.stdout}` : "",
        result.stderr ? `--- stderr ---\n${result.stderr}` : "",
      ]
        .filter(Boolean)
        .join("\n") || "(no output)"
    )}</pre>`;
  logEntry("result", summary);
}

function setIter(info) {
  if (iterInfoEl) iterInfoEl.textContent = info || "";
}

/* ---------------------------------------------------------------- *
 * Agent loop
 * ---------------------------------------------------------------- */

async function executeToolCall(call) {
  if (!window.basherTerminal) {
    throw new Error("Terminal panel is not initialized.");
  }
  // Mirror the command into the live terminal as a comment for visibility.
  window.basherTerminal
    .writeCommand(`# agent: ${call.command}`, { run: false })
    .catch(() => {});
  return window.basherTerminal.exec(call.command, { timeoutMs: TOOL_TIMEOUT_MS });
}

function shouldAutoRun(call, mode) {
  if (call.banned_reason) return false;
  if (mode === "autonomous") return call.danger === "safe";
  return false;
}

function approvalControls(call, mode, onDecision) {
  const banned = Boolean(call.banned_reason);
  const controls = [
    {
      label: banned ? "Banned" : "Run",
      onClick: (btn) => {
        if (banned) return;
        btn.disabled = true;
        onDecision("run");
      },
    },
    {
      label: "Skip",
      className: "secondary",
      onClick: (btn) => {
        btn.disabled = true;
        onDecision("skip");
      },
    },
    {
      label: "Edit",
      className: "secondary",
      onClick: () => {
        const next = window.prompt("Edit command before running:", call.command);
        if (next == null) return;
        call.command = String(next);
        onDecision("run-edited");
      },
    },
  ];
  if (banned) controls[0].className = "secondary";
  return controls;
}

async function approveCall(call, mode) {
  return new Promise((resolve) => {
    if (shouldAutoRun(call, mode)) {
      appendToolRequest(call, null);
      resolve("run");
      return;
    }
    appendToolRequest(call, approvalControls(call, mode, resolve));
  });
}

async function executeAndReport(call) {
  try {
    const result = await executeToolCall(call);
    appendToolResult(call, result);
    return {
      tool_call_id: call.id,
      command: call.command,
      stdout: result.stdout || "",
      stderr: result.stderr || "",
      exit_code: typeof result.code === "number" ? result.code : -1,
      timed_out: Boolean(result.timedOut),
    };
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    appendError(`Execution failed: ${msg}`);
    return {
      tool_call_id: call.id,
      command: call.command,
      stdout: "",
      stderr: `[client error] ${msg}`,
      exit_code: -1,
      timed_out: false,
    };
  }
}

async function driveAgentLoop(initialResponse) {
  let turn = initialResponse;
  const mode = modeEl.value;
  let maxIter = turn.max_iterations || 6;

  while (turn && !turn.done) {
    setIter(`iteration ${turn.iteration}/${maxIter}`);
    appendThoughts(turn.thoughts);

    const results = [];
    for (const call of turn.tool_calls || []) {
      const decision = await approveCall(call, mode);
      if (decision === "skip") {
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
      results.push(await executeAndReport(call));
    }

    if (!results.length) {
      appendError("No tool results were produced; halting loop.");
      break;
    }

    setStatus("Continuing agent reasoning...");
    turn = await window.basherAgentApi.submitToolResults({
      session_id: turn.session_id,
      results,
      debug: Boolean(debugEl.checked),
    });
  }

  if (turn) {
    setIter(`iteration ${turn.iteration}/${maxIter} • done`);
    appendThoughts(turn.thoughts);
    if (turn.final_answer) {
      responseEl.innerHTML = renderMarkdown(turn.final_answer);
      // Mirror final answer's commands as one-click chips.
      renderSuggestedCommands(extractCommands(turn.final_answer));
    } else {
      responseEl.textContent = "(no final answer)";
    }
    setStatus(`Done. Session: ${turn.session_id}`);
  }
}

/* ---------------------------------------------------------------- *
 * Suggested-commands chips (re-run a command from the final answer)
 * ---------------------------------------------------------------- */

function extractCommands(text) {
  const out = [];
  const src = String(text || "");
  const fenceRe = /```(bash|sh|shell|console|zsh)?\s*\n([\s\S]*?)```/gi;
  let m;
  while ((m = fenceRe.exec(src)) !== null) {
    for (const raw of (m[2] || "").split(/\r?\n/)) {
      const line = raw.replace(/^\s*[$#]\s+/, "").trim();
      if (!line || line.startsWith("//") || line.startsWith("# ")) continue;
      out.push(line);
    }
  }
  for (const raw of src.split(/\r?\n/)) {
    const line = raw.trim();
    if (/^\$\s+\S/.test(line)) out.push(line.replace(/^\$\s+/, ""));
  }
  const seen = new Set();
  return out.filter((c) => (!c || seen.has(c) ? false : (seen.add(c), true)));
}

function renderSuggestedCommands(commands) {
  if (!suggestedEl) return;
  suggestedEl.innerHTML = "";
  if (!commands.length) return;
  const header = document.createElement("div");
  header.style.fontSize = "12px";
  header.style.color = "#98a4b3";
  header.textContent = "Commands from the answer (click to run interactively):";
  suggestedEl.appendChild(header);
  for (const cmd of commands) {
    const row = document.createElement("div");
    row.className = "cmd-row";
    const code = document.createElement("code");
    code.textContent = cmd;
    row.appendChild(code);
    const runBtn = document.createElement("button");
    runBtn.textContent = "Run in terminal";
    runBtn.addEventListener("click", () => {
      window.basherTerminal
        ?.writeCommand(cmd, { run: true })
        .catch((err) => setStatus(`Terminal write failed: ${err.message || err}`));
    });
    row.appendChild(runBtn);
    suggestedEl.appendChild(row);
  }
}

/* ---------------------------------------------------------------- *
 * Top-level handlers
 * ---------------------------------------------------------------- */

async function runAgent() {
  const query = String(queryEl.value || "").trim();
  if (!query) {
    setStatus("Prompt is required.");
    return;
  }

  sendBtn.disabled = true;
  setStatus("Running agent...");
  responseEl.innerHTML = "";
  if (stepsEl) stepsEl.textContent = "";
  if (suggestedEl) suggestedEl.innerHTML = "";
  clearLog();
  appendSystem(`User: ${query}`);

  try {
    const payload = {
      query,
      mode: modeEl.value,
      debug: Boolean(debugEl.checked),
      base_model: Boolean(baseModelEl.checked),
      session_id: String(sessionIdEl.value || "").trim() || null,
    };
    const initial = await window.basherAgentApi.runAgent(payload);
    sessionIdEl.value = initial.session_id || sessionIdEl.value;
    if (stepsEl && initial.steps) stepsEl.textContent = asText(initial.steps);
    await driveAgentLoop(initial);
  } catch (err) {
    appendError(err && err.message ? err.message : String(err));
    setStatus(`Run failed: ${err.message || String(err)}`);
  } finally {
    sendBtn.disabled = false;
  }
}

async function loadSession() {
  const id = String(sessionIdEl.value || "").trim();
  if (!id) {
    setStatus("Session ID is required to load session.");
    return;
  }
  loadSessionBtn.disabled = true;
  setStatus(`Loading session ${id}...`);
  try {
    const out = await window.basherAgentApi.getSession(id);
    sessionEl.textContent = asText(out.messages || []);
    setStatus(
      `Session ${id} loaded (${out.iterations || 0} iterations, mode ${out.mode || "?"}).`
    );
  } catch (err) {
    setStatus(`Load failed: ${err.message || String(err)}`);
  } finally {
    loadSessionBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", runAgent);
loadSessionBtn.addEventListener("click", loadSession);
