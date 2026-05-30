"""
Agent loop for Basher.

Design notes
------------
The LLM we're driving (qwen2.5-coder:1.5b on local Ollama) is a *small* model —
1.5B parameters. That shapes every choice in this file:

  1. Single, tight system prompt. Long prose hurts small models (lost-in-the-
     middle + attention dilution).
  2. Strict JSON output contract enforced via Ollama's `format: "json"`. We
     never ask the model to free-form *and* structure at the same time.
  3. Two-three high-quality few-shot examples. Small models pattern-match;
     they don't reliably generalize from rules alone.
  4. XOR contract: each assistant turn either calls tools OR gives the final
     answer, never both. Eliminates a whole class of parse ambiguity.
  5. Hard cap on tools per turn (2) and total iterations per session (6).
     Prevents runaway loops and keeps each turn small enough to reason about.
  6. Server-side safety classifier in addition to the model's self-tag, so a
     model hallucinating `"danger": "safe"` for `rm -rf /` can't bypass us.
  7. Output validation + graceful fallback. If the model fails the schema,
     we extract a best-effort `final_answer` from the raw text instead of
     surfacing a 500.

The actual shell execution happens *in the desktop client* (Electron renderer
via node-pty), never on the API server. That keeps the server stateless
w.r.t. the user's machine and means the user can revoke a command before it
runs.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

import requests


OLLAMA_CHAT_URL_DEFAULT = "http://ollama:11434/api/chat"
OLLAMA_GENERATE_URL_DEFAULT = "http://ollama:11434/api/generate"

MAX_TOOL_CALLS_PER_TURN = 2
MAX_ITERATIONS = 6
MAX_TOOL_OUTPUT_CHARS = 4000  # what we feed back to the model per tool result


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# Engineering notes:
# - Persona first (one line). Models anchor on the first sentence.
# - Then capability (what tools exist).
# - Then output contract (exact JSON shape).
# - Then a *short* numbered policy. Long policies degrade small-model
#   adherence; we keep it to ~10 items.
# - We always show the schema verbatim — small models copy structure better
#   than they synthesize it from prose.
# - We *show* danger classification examples instead of describing them.
# - Encourage early termination: the model is biased to "keep working"; we
#   counter with explicit "stop and answer as soon as you can".

SYSTEM_PROMPT = """You are Basher, an expert Fedora Linux engineer with a real
bash terminal on the user's machine. The user runs Fedora 43.

You help with Fedora, bash, git, docker, systemd, dnf, SELinux, networking,
and developer workflows. You can either answer from knowledge + retrieved docs
OR run shell commands first to gather facts before answering.

# Tools

shell(command: string)
  Runs `bash -lc <command>` on the user's Fedora host.
  Returns: { stdout: string, stderr: string, exit_code: int, timed_out: bool }.
  Commands run non-interactively; never invoke editors or pagers.

# Output contract (return ONLY this JSON, no prose around it)

{
  "thoughts":     "<=2 short sentences explaining what you'll do next",
  "tool_calls":   [ { "type": "shell",
                      "command": "<one bash one-liner>",
                      "purpose": "<why you need this>",
                      "danger":  "safe" | "caution" | "destructive" } ],
  "final_answer": "<markdown answer for the user>" | null
}

# Rules

1. Exactly ONE of `tool_calls` (non-empty) OR `final_answer` (non-null) per
   turn. Never both. Never neither.
2. Use at most 2 tool_calls per turn. Read their output, then decide.
3. Prefer the smallest diagnostic that answers the question. `uname -r` over
   `dmesg | head -100`.
4. Stop calling tools as soon as you can answer. Most questions need 0–2 tools.
5. Danger classification:
   - safe:        read-only / diagnostic. `uname`, `rpm -qa`, `systemctl status`,
                  `ls`, `cat`, `journalctl --no-pager -n 50`, `dnf info`.
   - caution:     reversible state change. `sudo dnf install`, `systemctl
                  restart|reload`, `git pull`, `firewall-cmd --reload`.
   - destructive: irreversible. `rm -rf`, `dd`, `mkfs`, `dnf remove kernel*`,
                  `rpm -e --nodeps`. Only if the user explicitly asked. Explain
                  the risk in `thoughts`.
6. Never run interactive editors (`vim`, `nano`) or full-screen TUIs (`htop`,
   `top`, `less` without `-F`). Use `cat`, `head`, `tail`, `journalctl
   --no-pager`.
7. Quote paths containing spaces. Use absolute paths. Don't rely on `cd`
   persisting between calls — each call is a fresh `bash -lc`.
8. Cite the Documentation block when it is relevant. Do not invent flags or
   file paths that aren't in docs or shell output.
9. Answer in the user's language. If they wrote Finnish, answer Finnish.
10. When you give `final_answer`, be concise: short prose + a fenced code
    block of the exact commands the user should run. No filler.

# Examples

User: "Mikä kernel mulla on käytössä?"
Assistant:
{"thoughts":"Tarkistetaan käytössä oleva kernel uname:lla.","tool_calls":[{"type":"shell","command":"uname -r","purpose":"running kernel version","danger":"safe"}],"final_answer":null}

Tool result (uname -r): "6.14.6-300.fc43.x86_64"
Assistant:
{"thoughts":"Saatu vastaus, voin vastata.","tool_calls":[],"final_answer":"Käytössä on **6.14.6-300.fc43.x86_64** (Fedora 43).\\n\\n```bash\\nuname -r\\n```"}

User: "How do I check which Fedora release I'm on?"
Assistant:
{"thoughts":"Standard one-liner; I can answer without running anything.","tool_calls":[],"final_answer":"On Fedora, read `/etc/os-release` or use `hostnamectl`:\\n\\n```bash\\ncat /etc/os-release\\nhostnamectl\\n```"}

User: "Is podman running?"
Assistant:
{"thoughts":"Need to query systemd for the podman.socket and podman.service state.","tool_calls":[{"type":"shell","command":"systemctl --user is-active podman.socket podman.service 2>&1; systemctl is-active podman.service 2>&1","purpose":"check podman user+system service state","danger":"safe"}],"final_answer":null}
"""


# ---------------------------------------------------------------------------
# Safety classifier
# ---------------------------------------------------------------------------

_DESTRUCTIVE_RE = re.compile(
    r"""(
        \brm\s+-[a-zA-Z]*r[a-zA-Z]*f?[a-zA-Z]*\s+/(?:\s|$)        # rm -rf /
      | \brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+--no-preserve-root
      | \bdd\s+[^|]*\bof=/dev/[sh]d
      | \bmkfs(\.\w+)?\s+/dev/
      | \bshred\b
      | :\(\)\s*\{[^}]*\}\s*;:                                    # fork bomb
      | >\s*/dev/sd[a-z]
      | \bdnf\s+(remove|erase)\s+.*kernel
      | \brpm\s+-e\s+--nodeps\b
      | \bchattr\s+-i\b.*/
      | \bparted\b.*\bmklabel\b
      | \bcryptsetup\s+luksFormat\b
      | \bgit\s+push\s+--force\b.*\b(main|master)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_CAUTION_RE = re.compile(
    r"""(
        \bsudo\b
      | \bdnf\s+(install|remove|erase|update|upgrade|downgrade|reinstall|autoremove)\b
      | \brpm\s+-(i|e|U)\b
      | \bflatpak\s+(install|uninstall|update)\b
      | \bsystemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask|edit)\b
      | \bjournalctl\s+--vacuum
      | \bfirewall-cmd\b.*(--add|--remove|--reload|--permanent)
      | \bsemanage\b | \bsetsebool\b | \bsetenforce\b
      | \bchmod\s+(-R|[0-7]{3,4})
      | \bchown\s+-R
      | \bmount\b | \bumount\b
      | \bgit\s+(reset\s+--hard|clean\s+-[fdx]|push)\b
      | \bdocker\s+(rm|rmi|prune|stop|kill|system\s+prune)\b
      | \bpodman\s+(rm|rmi|prune|stop|kill|system\s+prune)\b
      | \busermod\b | \buseradd\b | \buserdel\b | \bgroupmod\b
      | >\s*/etc/
      | \btee\s+(-a\s+)?/etc/
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def classify_command(command: str) -> str:
    """Independent safety classification — never trusts model self-tag."""
    cmd = (command or "").strip()
    if not cmd:
        return "safe"
    if _DESTRUCTIVE_RE.search(cmd):
        return "destructive"
    if _CAUTION_RE.search(cmd):
        return "caution"
    return "safe"


_BANNED_RE = re.compile(
    r"""(
        \bvim\b | \bvi\b | \bnano\b | \bemacs\b
      | \btop\b(?!\s*-) | \bhtop\b | \bbtop\b
      | \bless\b(?!\s+-F)
      | \bcurl\s+.*\|\s*(sudo\s+)?(bash|sh)\b
      | \bwget\s+.*\|\s*(sudo\s+)?(bash|sh)\b
      | \brm\s+-[a-zA-Z]*\s+/\s*$
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def is_banned(command: str) -> Optional[str]:
    """Return a reason if the command should be refused outright, else None."""
    if _BANNED_RE.search(command or ""):
        return (
            "Interactive editors / TUI tools / piped-to-shell installs are not "
            "allowed from the agent. Run them yourself in the terminal."
        )
    return None


# ---------------------------------------------------------------------------
# Output parsing & validation
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    # Strip ```json fences if the model added them despite format=json.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Greedy: pull the first balanced { ... } that parses.
    start = raw.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = raw.find("{", start + 1)
    return None


def _normalize_response(raw: str) -> dict:
    """Validate model output against the contract. Always returns a valid shape."""
    parsed = _try_parse_json(raw)

    if not isinstance(parsed, dict):
        # Schema-failure fallback: treat the raw text as the final answer so
        # we don't dead-end the user.
        return {
            "thoughts": "(model returned non-JSON, falling back to plain text)",
            "tool_calls": [],
            "final_answer": (raw or "").strip() or "(no response)",
            "_schema_ok": False,
        }

    thoughts = str(parsed.get("thoughts") or "").strip()
    raw_calls = parsed.get("tool_calls") or []
    final_answer = parsed.get("final_answer")

    if not isinstance(raw_calls, list):
        raw_calls = []

    tool_calls = []
    for c in raw_calls[:MAX_TOOL_CALLS_PER_TURN]:
        if not isinstance(c, dict):
            continue
        command = str(c.get("command") or "").strip()
        if not command:
            continue
        model_danger = str(c.get("danger") or "safe").lower()
        server_danger = classify_command(command)
        # Always escalate, never downgrade.
        danger_rank = {"safe": 0, "caution": 1, "destructive": 2}
        danger = (
            server_danger
            if danger_rank.get(server_danger, 0) >= danger_rank.get(model_danger, 0)
            else model_danger
        )
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": str(c.get("type") or "shell").lower() or "shell",
            "command": command,
            "purpose": str(c.get("purpose") or "").strip(),
            "danger": danger,
            "model_danger": model_danger,
            "banned_reason": is_banned(command),
        })

    has_calls = bool(tool_calls)
    has_answer = isinstance(final_answer, str) and final_answer.strip() != ""

    # XOR fix-up: if model violated the contract, prefer the side with content.
    if has_calls and has_answer:
        final_answer = None
    elif not has_calls and not has_answer:
        final_answer = thoughts or "(no answer)"

    return {
        "thoughts": thoughts,
        "tool_calls": tool_calls,
        "final_answer": final_answer if not tool_calls else None,
        "_schema_ok": True,
    }


# ---------------------------------------------------------------------------
# Ollama chat call
# ---------------------------------------------------------------------------


def _ollama_chat(
    messages: list[dict],
    *,
    model: str,
    url: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_predict: int = 768,
    timeout: int = 120,
    force_json: bool = True,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "repeat_penalty": 1.05,
        },
    }
    if force_json:
        payload["format"] = "json"

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # /api/chat returns { "message": { "role": "assistant", "content": "..." } }
    msg = data.get("message") or {}
    return str(msg.get("content") or "")


def _ollama_chat_with_fallback(
    messages: list[dict],
    *,
    chat_url: str,
    generate_url: str,
    model: str,
    **kwargs,
) -> str:
    """Try /api/chat first; if Ollama is too old, fall back to /api/generate
    with a flattened transcript."""
    try:
        return _ollama_chat(messages, model=model, url=chat_url, **kwargs)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code != 404:
            raise
    except requests.ConnectionError:
        raise

    flat = []
    for m in messages:
        role = m.get("role", "user").upper()
        flat.append(f"<|{role}|>\n{m.get('content','')}")
    flat.append("<|ASSISTANT|>\n")
    payload = {
        "model": model,
        "prompt": "\n\n".join(flat),
        "stream": False,
        "format": "json" if kwargs.get("force_json", True) else None,
        "options": {
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.9),
            "num_predict": kwargs.get("num_predict", 768),
            "repeat_penalty": 1.05,
        },
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    r = requests.post(generate_url, json=payload, timeout=kwargs.get("timeout", 120))
    r.raise_for_status()
    return str(r.json().get("response") or "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_initial_messages(
    user_query: str,
    *,
    documentation: str = "",
    base_model: bool = False,
) -> list[dict]:
    """Build the message array for the first agent turn."""
    sys_content = SYSTEM_PROMPT
    if not base_model and documentation:
        sys_content += (
            "\n\n# Documentation (retrieved for this question)\n"
            "Use only when relevant. Do not fabricate paths or flags.\n\n"
            + documentation
        )
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_query.strip()},
    ]


def truncate_for_history(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    s = text or ""
    if len(s) <= limit:
        return s
    head = s[: limit // 2]
    tail = s[-limit // 2 :]
    return f"{head}\n... [{len(s) - limit} chars truncated] ...\n{tail}"


def format_tool_result_message(
    tool_call_id: str, command: str, result: dict
) -> dict:
    """Render a tool result back into the chat history.

    The role is `user` (Ollama doesn't always honour `tool` role for small
    models), but the content is clearly delimited so the model parses it.
    """
    stdout = truncate_for_history(str(result.get("stdout", "")))
    stderr = truncate_for_history(str(result.get("stderr", "")))
    body = (
        f"[tool_result id={tool_call_id} command={json.dumps(command)}]\n"
        f"exit_code: {result.get('exit_code', result.get('code', 'unknown'))}\n"
        f"timed_out: {bool(result.get('timed_out', False))}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}\n"
        f"[/tool_result]\n\n"
        f"Reason about the result and either call another tool (max 2) or "
        f"give the final_answer. Reply with the JSON object only."
    )
    return {"role": "user", "content": body}


def run_agent_turn(
    messages: list[dict],
    *,
    chat_url: str,
    generate_url: str,
    model: str,
) -> dict:
    """Run one assistant turn. Returns the normalized response dict plus the
    raw model output for debugging."""
    raw = _ollama_chat_with_fallback(
        messages,
        chat_url=chat_url,
        generate_url=generate_url,
        model=model,
    )
    norm = _normalize_response(raw)
    norm["raw"] = raw
    return norm


def append_assistant_turn(messages: list[dict], norm: dict) -> None:
    """Append the assistant's turn to history in the canonical JSON form, so
    the next turn sees a consistent transcript."""
    payload = {
        "thoughts": norm.get("thoughts", ""),
        "tool_calls": [
            {
                "id": c["id"],
                "type": c["type"],
                "command": c["command"],
                "purpose": c.get("purpose", ""),
                "danger": c["danger"],
            }
            for c in norm.get("tool_calls", [])
        ],
        "final_answer": norm.get("final_answer"),
    }
    messages.append({"role": "assistant", "content": json.dumps(payload)})


__all__ = [
    "SYSTEM_PROMPT",
    "MAX_TOOL_CALLS_PER_TURN",
    "MAX_ITERATIONS",
    "OLLAMA_CHAT_URL_DEFAULT",
    "OLLAMA_GENERATE_URL_DEFAULT",
    "classify_command",
    "is_banned",
    "build_initial_messages",
    "format_tool_result_message",
    "run_agent_turn",
    "append_assistant_turn",
    "truncate_for_history",
]
