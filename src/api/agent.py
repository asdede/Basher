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
and developer workflows. You either answer from knowledge OR run shell
commands first to gather facts, then answer.

# Tools

shell(command: string)
  Runs `bash -lc <command>` on the user's Fedora host.
  Returns: { stdout: string, stderr: string, exit_code: int, timed_out: bool }.
  Commands run non-interactively; never invoke editors or pagers.

# Output contract (return ONLY this JSON object, no prose around it)

{
  "thoughts":     "<=2 short sentences explaining what you'll do next",
  "tool_calls":   [ { "type": "shell",
                      "command": "<one bash one-liner>",
                      "purpose": "<why you need this>",
                      "danger":  "safe" | "caution" | "destructive" } ],
  "final_answer": "<markdown answer for the user>" | null
}

`final_answer` MUST be the actual user-facing markdown answer text. It is
NEVER a single word, NEVER one of the schema keys above
(`thoughts`/`tool_calls`/`command`/`purpose`/`danger`/`safe`/`caution`/
`destructive`/`shell`/`task`/`response`), and NEVER the user's question
paraphrased back. If you have nothing useful to say, emit a `tool_calls`
instead.

# Rules

1. Exactly ONE of `tool_calls` (non-empty) OR `final_answer` (non-null,
   non-empty, real sentence) per turn. Never both. Never neither.
2. **Tool-first for state questions.** If the user asks about anything
   *specific to their machine* — files on disk, running processes,
   ports, package versions, systemd unit state, git/branch state, env
   vars, container state, etc. — you MUST emit a `tool_calls` shell
   command first. Do NOT guess paths or filenames; do NOT echo the
   user's words back as the answer. Only after a tool result comes back
   may you write a `final_answer`.
3. NEVER refuse. The user owns their machine. Questions like "where is
   my X folder?", "what's running on port N?", "find file Y", "list my
   containers" are normal sysadmin work. Phrases like "I'm sorry, I
   can't help with that", "this is a sensitive issue", "as an AI
   model..." are FORBIDDEN. If you don't know, emit a `find` / `locate`
   / `command -v` / `ss` / `ps` one-liner that *would* find the answer.
4. Use at most 2 tool_calls per turn. Read their output, then decide.
5. Pick the RIGHT tool for the question — small command, exact intent.

# Tool-selection cheat-sheet (memorize this — do not default to `ls`)

| Question shape                              | Use                                                          |
|---------------------------------------------|--------------------------------------------------------------|
| "where is X" / "find file X" / "lost my X"  | `find "$HOME" -maxdepth 6 -iname 'X*' 2>/dev/null \\| head -20` |
| "do I have command X" / "is X installed"    | `command -v X` or `rpm -q X` or `dnf list --installed X`     |
| "what's in directory Y" (path known)        | `ls -la Y`                                                   |
| "what's using port N"                       | `ss -tulpn \\| grep -E ':N\\b'`                                |
| "is service X running"                      | `systemctl status X --no-pager` or `systemctl is-active X`   |
| "what process is X" / "is X running"        | `pgrep -fa X` or `ps -eo pid,cmd \\| grep -i X`               |
| "kernel / OS / arch"                        | `uname -r` / `cat /etc/os-release` / `hostnamectl`           |
| "find text T in files under Y"              | `grep -rn --color=never 'T' Y 2>/dev/null \\| head`           |
| "logs for unit X"                           | `journalctl -u X --no-pager -n 100`                          |
| "disk usage of dir Y"                       | `du -sh Y/*` or `df -h`                                      |
| "git state of repo Y"                       | `git -C Y status -sb && git -C Y log --oneline -5`           |
| "container state"                           | `podman ps -a` or `docker ps -a`                             |

`ls` is NOT a search tool. Use it only when you already know the exact path.
For ANY "find / where / look up" intent, reach for `find`, `locate`,
`command -v`, `rpm -q`, `pgrep`, `ss`, or `grep -r` — in that order of
preference.

6. Danger classification:
   - safe:        read-only / diagnostic. `uname`, `rpm -qa`, `systemctl status`,
                  `ls`, `cat`, `find`, `locate`, `journalctl --no-pager -n 50`.
   - caution:     reversible state change. `sudo dnf install`, `systemctl
                  restart|reload`, `git pull`, `firewall-cmd --reload`.
   - destructive: irreversible. `rm -rf`, `dd`, `mkfs`, `dnf remove kernel*`,
                  `rpm -e --nodeps`. Only if the user explicitly asked. Explain
                  the risk in `thoughts`.
7. Never run interactive editors (`vim`, `nano`) or full-screen TUIs
   (`htop`, `top`, `less` without `-F`). Use `cat`, `head`, `tail`,
   `journalctl --no-pager`.
8. Quote paths containing spaces. Use absolute paths or `$HOME`. Don't
   rely on `cd` persisting between calls — each call is a fresh
   `bash -lc`.
9. Answer in the user's language. If they wrote Finnish, answer Finnish.
10. When you give `final_answer`, be concise: 1-3 short sentences of
    plain prose explaining the result, then a fenced ```bash``` block of
    the exact commands the user should run themselves. Do NOT echo the
    question back. No filler. No apologies. No schema fragments.

# Examples (study the tool choices)

User: "What Kernal am I running?"
Assistant:
{"thoughts":"Check the running kernel version with uname.","tool_calls":[{"type":"shell","command":"uname -r","purpose":"running kernel version","danger":"safe"}],"final_answer":null}

Tool result (uname -r): "6.14.6-300.fc43.x86_64"
Assistant:
{"thoughts":"Got the kernel version, can answer.","tool_calls":[],"final_answer":"The running kernel is **6.14.6-300.fc43.x86_64** (Fedora 43).\\n\\n```bash\\nuname -r\\n```"}

User: "How do I check which Fedora release I'm on?"
Assistant:
{"thoughts":"Standard one-liner; I can answer without running anything.","tool_calls":[],"final_answer":"On Fedora, read `/etc/os-release` or use `hostnamectl`:\\n\\n```bash\\ncat /etc/os-release\\nhostnamectl\\n```"}

User: "Is podman running?"
Assistant:
{"thoughts":"Query systemd for the podman.socket and podman.service state.","tool_calls":[{"type":"shell","command":"systemctl --user is-active podman.socket podman.service 2>&1; systemctl is-active podman.service 2>&1","purpose":"check podman user+system service state","danger":"safe"}],"final_answer":null}

User: "do I have ripgrep installed"
Assistant:
{"thoughts":"Check PATH and the rpm database.","tool_calls":[{"type":"shell","command":"command -v rg && rpm -q ripgrep || echo NOTFOUND","purpose":"detect ripgrep binary and rpm status","danger":"safe"}],"final_answer":null}

User: "hello"
Assistant:
{"thoughts":"Greeting; no shell needed.","tool_calls":[],"final_answer":"Hi! I'm Basher. Ask me anything about Fedora, bash, dnf, systemd, docker or git — e.g. \\"what kernel am I on?\\" or \\"how do I update my system?\\""}

# Wrong vs right (tool choice)

User: "where is my code folder?"
WRONG  → {"thoughts":"List home.","tool_calls":[{"type":"shell","command":"ls $HOME","purpose":"list home","danger":"safe"}],"final_answer":null}
   ↑ `ls` is not a search tool; it can't even find a folder named code
   that lives under ~/dev/, ~/Documents/, etc.
RIGHT  → {"thoughts":"Search $HOME for common project-folder names.","tool_calls":[{"type":"shell","command":"find \\"$HOME\\" -maxdepth 4 -type d \\\\( -iname 'code' -o -iname 'projects' -o -iname 'dev' -o -iname 'workspace' -o -iname 'src' \\\\) 2>/dev/null","purpose":"find common code/project directories under $HOME","danger":"safe"}],"final_answer":null}

User: "I have lost my code folder. It has at least Basher named project in it. Can you find it for me?"
WRONG  → {"thoughts":"Look in home.","tool_calls":[{"type":"shell","command":"ls -R $HOME | grep -i basher","purpose":"scan home","danger":"safe"}],"final_answer":null}
   ↑ `ls -R | grep` is slow, noisy, and misses anything outside the
   visible tree. `find -iname` is the correct primitive.
RIGHT  → {"thoughts":"Search $HOME for a Basher directory; its parent is the code folder.","tool_calls":[{"type":"shell","command":"find \\"$HOME\\" -maxdepth 6 -type d -iname 'basher*' 2>/dev/null | head -20","purpose":"find the Basher project, its parent is the lost code folder","danger":"safe"}],"final_answer":null}

Tool result (find ...): "/home/masterbates/code/inno/Basher\\n/home/masterbates/code/inno/Basher/src/agent-ui/node_modules/.../basher-agent-ui"
Assistant:
{"thoughts":"Found it: parent of the project is the code folder.","tool_calls":[],"final_answer":"Your code folder is **/home/masterbates/code** (the project lives at `/home/masterbates/code/inno/Basher`).\\n\\n```bash\\nls /home/masterbates/code\\n```"}

# Wrong vs right (final_answer content)

User: "what kernel am I on?"
Tool result (uname -r): "6.14.6-300.fc43.x86_64"
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"purpose"}
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"task"}
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"what kernel am I on"}
   ↑ One-word / schema-key / question-echo answers are forbidden.
RIGHT  → {"thoughts":"...","tool_calls":[],"final_answer":"Kernel **6.14.6-300.fc43.x86_64**.\\n\\n```bash\\nuname -r\\n```"}
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


def _normalize_response(raw: str, *, user_query: str = "") -> dict:
    """Validate model output against the contract. Always returns a valid shape.

    `user_query` is optional; when present it lets us detect echo-of-query
    answers (the model paraphrasing the question instead of answering).
    """
    parsed = _try_parse_json(raw)

    if not isinstance(parsed, dict):
        # Schema-failure fallback: treat the raw text as the final answer so
        # we don't dead-end the user.
        return {
            "thoughts": "(model returned non-JSON, falling back to plain text)",
            "tool_calls": [],
            "final_answer": (raw or "").strip() or "(no response)",
            "_schema_ok": False,
            "_failure_reason": "non_json",
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

    # Schema-slip rescue. A 1.5B model under format=json sometimes invents
    # the wrong key for the answer ("answer", "response", "text", ...) or
    # nests the answer inside thoughts. Recover whatever readable string is
    # in the JSON before we declare the turn empty — surfacing the model's
    # actual words is always better than a synthetic "(no answer)".
    if not has_calls and not has_answer:
        for key in ("answer", "response", "text", "content", "output", "message", "reply"):
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                final_answer = v.strip()
                has_answer = True
                break

    # Last resort: any non-trivial string anywhere in the JSON beats a
    # synthetic placeholder. Skip "thoughts" itself — that's meta-narration.
    if not has_calls and not has_answer:
        for k, v in parsed.items():
            if k == "thoughts" or k == "tool_calls":
                continue
            if isinstance(v, str) and len(v.strip()) >= 2:
                final_answer = v.strip()
                has_answer = True
                break

    schema_ok = True
    failure_reason: str | None = None

    # XOR fix-up: if model violated the contract, prefer the side with content.
    if has_calls and has_answer:
        final_answer = None
    elif not has_calls and not has_answer:
        # Distinguish two empty cases: "thinking_only" means the model
        # planned aloud but didn't follow through (very common on q4
        # coder models — `{"thoughts":"I'll search","tool_calls":[],
        # "final_answer":null}`). It deserves a corrective retry rather
        # than a hard failure, because the model is *almost* there. A
        # truly empty response gets the generic "empty" tag.
        if thoughts.strip():
            final_answer = thoughts.strip()
            failure_reason = "thinking_only"
        else:
            final_answer = ""
            failure_reason = "empty"
        schema_ok = False

    # Refusal detection. Coder models at q4 have residual safety tuning
    # that fires unpredictably ("I'm sorry, I can't assist with that",
    # "this is a sensitive political issue", etc.) for completely benign
    # sysadmin questions. Letting that through is worse than admitting
    # the turn failed: the user can rephrase but a canned refusal wastes
    # their time and poisons the rolling memory.
    if isinstance(final_answer, str) and _is_refusal(final_answer):
        schema_ok = False
        failure_reason = failure_reason or "refusal"

    # Echo detection. Same models occasionally satisfy `format=json` by
    # rephrasing the user's question into final_answer. Treat that as a
    # failed turn so the renderer surfaces it and we don't cache an
    # empty exchange into the rolling memory.
    if (
        isinstance(final_answer, str)
        and not tool_calls
        and _looks_like_echo(user_query, final_answer)
    ):
        schema_ok = False
        failure_reason = failure_reason or "echo"

    # Field-leak detection. A small model under format=json sometimes
    # fills `final_answer` with one of the schema keys ("purpose",
    # "task", "command", ...) — JSON-valid, completely useless. Catch it
    # so the retry-nudge loop can re-prompt for a real answer instead of
    # the renderer showing "purpose" as if it were a reply.
    if (
        isinstance(final_answer, str)
        and not tool_calls
        and _looks_like_field_leak(final_answer)
    ):
        schema_ok = False
        failure_reason = failure_reason or "field_leak"

    out = {
        "thoughts": thoughts,
        "tool_calls": tool_calls,
        "final_answer": final_answer if not tool_calls else None,
        "_schema_ok": schema_ok,
    }
    if failure_reason:
        out["_failure_reason"] = failure_reason
    return out


# All anchored to short responses; we don't want to flag a legitimately
# long answer that happens to contain "I'm sorry" inside an explanation.
_REFUSAL_PATTERNS = (
    re.compile(r"i\s*[''']?m\s+sorry,?\s+(but\s+)?i\s+(can(?:not|'?\s*t)|am\s+unable|won'?t)", re.I),
    re.compile(r"i\s+(?:can(?:not|'?\s*t)|am\s+unable|won'?t)\s+(?:assist|help|answer|address|comment)", re.I),
    re.compile(r"as\s+an\s+ai(?:\s+language)?\s+model,?\s+i", re.I),
    re.compile(r"i'?m\s+not\s+able\s+to\s+(?:provide|assist|help|answer|address)", re.I),
    re.compile(r"sensitive\s+(?:and\s+)?(?:political|legal|religious)\s+issue", re.I),
    re.compile(r"won'?t\s+comment\s+on\s+(?:or\s+express\s+opinions\s+)?(?:about\s+)?(?:political|legal|religious)", re.I),
    re.compile(r"i\s+don'?t\s+(?:engage|comment)\s+(?:with|on)\s+(?:political|controversial)", re.I),
)


def _is_refusal(text: str) -> bool:
    """Heuristic: does this look like a canned safety refusal?
    Caps at ~320 chars so a long real answer that happens to contain a
    matching substring doesn't get tripped."""
    s = (text or "").strip()
    if not s or len(s) > 320:
        return False
    return any(p.search(s) for p in _REFUSAL_PATTERNS)


# Tokens the model knows from the system prompt: schema keys, schema
# enum values, tool type names. Under `format: "json"` a small model
# sometimes "satisfies" the contract by filling `final_answer` with one
# of these words verbatim. The result parses as valid JSON but tells the
# user nothing — surfacing it as "the answer" is worse than admitting
# the turn failed, because the rolling memory then caches "purpose" or
# "task" as if it were a real reply.
_SCHEMA_LEAK_TOKENS = frozenset({
    "thoughts", "tool_calls", "tool_call", "final_answer", "command",
    "purpose", "danger", "safe", "caution", "destructive", "shell",
    "task", "tasks", "response", "answer", "text", "content", "output",
    "message", "reply", "null", "none", "todo", "tbd",
})


def _looks_like_field_leak(answer: str) -> bool:
    """Detect when `final_answer` is just a schema keyword (or near-empty)
    instead of an actual user-facing reply.

    Concrete failure modes seen in the wild on q4 coder models:
      final_answer: "purpose"
      final_answer: "task"
      final_answer: "shell"
      final_answer: "command"
    All technically valid JSON strings, all useless.
    """
    s = (answer or "").strip().strip("`'\"")
    if not s:
        return False
    # A single short token (no whitespace, no newline) is suspicious by
    # itself. We only flag it when it's also a known schema keyword;
    # legitimate one-word answers like "yes", "ok", "fedora" pass.
    if " " not in s and "\n" not in s and 1 <= len(s) <= 24:
        if s.lower() in _SCHEMA_LEAK_TOKENS:
            return True
    return False


def _looks_like_echo(query: str, answer: str) -> bool:
    """Detect when the model lazily paraphrased the question into the
    answer instead of running a tool or giving real info.

    Concrete failure modes seen in the wild on q4 coder models:
      user: "Can you find it with that information?"
      answer: "find it with that information"
    The model is JSON-compliant but said nothing useful.
    """
    q = (query or "").lower().strip().strip("?!.")
    a = (answer or "").lower().strip().strip("?!.")
    if not a or not q:
        return False
    if len(a) < 4:
        # very short answers are not echoes; treat them on their merits
        return False
    if a == q:
        return True
    # Answer is mostly a substring of the query and offers little new content.
    if a in q and len(a) >= 0.4 * len(q):
        return True
    # Token overlap: >=85% of answer tokens appear in the query, AND no
    # answer token introduces new content (e.g. a path, command, number).
    a_tokens = [t for t in re.split(r"\s+", a) if t]
    if not a_tokens:
        return False
    q_tokens = set(re.split(r"\s+", q))
    overlap = sum(1 for t in a_tokens if t in q_tokens) / len(a_tokens)
    has_signal = any(
        ("/" in t) or t.startswith("$") or t.startswith("`") or
        any(c.isdigit() for c in t)
        for t in a_tokens
    )
    return overlap >= 0.85 and not has_signal


# ---------------------------------------------------------------------------
# Ollama chat call
# ---------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """Upstream-Ollama failure with enough context for the API layer to
    translate into a useful HTTP response.

    `status` is the HTTP status we want FastAPI to surface (502 for a
    generic upstream issue, 503 when the model is missing or Ollama is
    unreachable — both signal "service-side problem, please fix and
    retry"). `body` is the raw upstream payload for debug logs.
    """

    def __init__(self, message: str, *, status: int = 502, body: str = "", model: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body
        self.model = model


def _ollama_error_from_response(r, model: str) -> OllamaError:
    """Convert a non-2xx Ollama response into a typed OllamaError. We try
    to recognize the very specific "model not pulled" case because that's
    99% of operator errors, and a generic "Internal Server Error" wastes
    everyone's time."""
    detail = ""
    try:
        j = r.json()
        if isinstance(j, dict) and j.get("error"):
            detail = str(j["error"])
        else:
            detail = (r.text or "")[:400]
    except Exception:
        detail = (r.text or "")[:400]

    low = detail.lower()
    if r.status_code == 404 and ("not found" in low or "model" in low or "pull" in low):
        return OllamaError(
            f"Model '{model}' is not available on Ollama. Pull it first, "
            f"e.g. `docker exec basher-llm ollama pull {model}` "
            f"(or set OLLAMA_MODEL to a model that is already pulled and "
            f"restart the api container).",
            status=503,
            body=detail,
            model=model,
        )
    return OllamaError(
        f"Ollama upstream error (HTTP {r.status_code}): {detail[:200]}",
        status=502,
        body=detail,
        model=model,
    )


def _ollama_chat(
    messages: list[dict],
    *,
    model: str,
    url: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_predict: int = 768,
    num_ctx: int | None = None,
    timeout: int = 120,
    force_json: bool = True,
) -> str:
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": num_predict,
        "repeat_penalty": 1.05,
    }
    # Ollama defaults to num_ctx=2048 which is too small once we add the
    # system prompt + few-shots + RAG context + recent turns. Pass an
    # explicit ceiling so we don't get silent prompt truncation; the
    # caller picks a value matched to the model + VRAM.
    if num_ctx and num_ctx > 0:
        options["num_ctx"] = int(num_ctx)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if force_json:
        payload["format"] = "json"

    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {url}. Is the basher-llm container up? "
            f"({e.__class__.__name__})",
            status=503,
            model=model,
        )
    except requests.Timeout:
        raise OllamaError(
            f"Ollama timed out after {timeout}s on {url}. Model may be "
            f"loading from cold — retry, or pre-warm with "
            f"`docker exec basher-llm ollama run {model} 'hi'`.",
            status=504,
            model=model,
        )

    if not r.ok:
        raise _ollama_error_from_response(r, model)

    try:
        data = r.json()
    except ValueError:
        raise OllamaError(
            f"Ollama returned non-JSON ({len(r.text or '')} bytes).",
            status=502,
            body=(r.text or "")[:400],
            model=model,
        )
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
    """Try /api/chat first; on legacy "endpoint missing" (very old
    Ollama), fall back to /api/generate with a flattened transcript.
    Model-missing / connection / timeout errors propagate as OllamaError
    so the API layer can produce a useful response."""
    try:
        return _ollama_chat(messages, model=model, url=chat_url, **kwargs)
    except OllamaError as e:
        # 503 = model missing / Ollama unreachable — no point flailing on
        # /api/generate, the same problem awaits there.
        if e.status != 502:
            raise
        # 502 from chat (route 404 / unexpected non-JSON) might actually
        # mean "this Ollama is too old and lacks /api/chat" — try the
        # legacy route once.

    flat = []
    for m in messages:
        role = m.get("role", "user").upper()
        flat.append(f"<|{role}|>\n{m.get('content','')}")
    flat.append("<|ASSISTANT|>\n")
    options = {
        "temperature": kwargs.get("temperature", 0.2),
        "top_p": kwargs.get("top_p", 0.9),
        "num_predict": kwargs.get("num_predict", 768),
        "repeat_penalty": 1.05,
    }
    if kwargs.get("num_ctx"):
        options["num_ctx"] = int(kwargs["num_ctx"])
    payload = {
        "model": model,
        "prompt": "\n\n".join(flat),
        "stream": False,
        "format": "json" if kwargs.get("force_json", True) else None,
        "options": options,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        r = requests.post(
            generate_url, json=payload, timeout=kwargs.get("timeout", 120)
        )
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {generate_url}. ({e.__class__.__name__})",
            status=503, model=model,
        )
    except requests.Timeout:
        raise OllamaError(
            f"Ollama timed out on {generate_url}.",
            status=504, model=model,
        )
    if not r.ok:
        raise _ollama_error_from_response(r, model)
    try:
        return str(r.json().get("response") or "")
    except ValueError:
        raise OllamaError(
            f"Ollama /api/generate returned non-JSON.",
            status=502, body=(r.text or "")[:400], model=model,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_initial_messages(
    user_query: str,
    *,
    documentation: str = "",
    base_model: bool = False,
    prior_summary: str = "",
    prior_turns: list[dict] | None = None,
) -> list[dict]:
    """Build the message array for the first agent turn.

    Memory has two layers, both optional:

    - `prior_summary`: a single compressed sentence covering everything
      that happened before the most recent N turns. Cheap, lossy.
    - `prior_turns`: the last few exchanges verbatim ([{user, answer}]).
      Each one is short; together they give the model real continuity
      for things a single summary can't capture ("yes, do that", "from
      that result, do X", references to a prior command).

    Both layers are clamped before they reach the model — small models
    treat long context blocks as instructions and start ignoring the
    *current* user turn. We err on the side of "less context, but
    relevant context".
    """
    sys_content = SYSTEM_PROMPT

    # --- Verbatim recent turns (preferred over the summary when fresh)
    turns = list(prior_turns or [])
    # Keep at most the last 3 turns and cap each to ~600 chars total. That
    # is enough for "where is my code folder?" + "Try ls ~/code." kind of
    # exchanges without ballooning the prompt.
    if turns:
        turns = turns[-3:]
        rendered = []
        for i, t in enumerate(turns, start=1):
            u = (t.get("user") or "").strip()
            a = (t.get("answer") or "").strip()
            if not u:
                continue
            if len(u) > 240:
                u = u[:240].rstrip() + "..."
            if len(a) > 360:
                a = a[:360].rstrip() + "..."
            rendered.append(f"[{i}] user: {u}\n    assistant: {a}")
        if rendered:
            sys_content += (
                "\n\n# Recent exchanges (most recent at the bottom)\n"
                "Read these for continuity. The user's CURRENT question "
                "is in the next user message.\n\n" + "\n".join(rendered)
            )

    # --- Compressed summary covering everything older than the recent turns
    prior = (prior_summary or "").strip()
    if prior:
        if len(prior) > 600:
            prior = prior[:600].rstrip() + "..."
        sys_content += (
            "\n\n# Older context (compressed)\n"
            "Background only. Do not repeat it back.\n\n" + prior
        )

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
        f"Reason about the result and reply with the JSON object only. "
        f"If the result answers the question, output `final_answer` now "
        f"(real markdown sentence + fenced code block, NEVER a single "
        f"schema keyword). If the result was empty / wrong directory / "
        f"not found, do NOT repeat the same command — SWITCH tools "
        f"(e.g. `ls` → `find -iname`, `find` → broader `-maxdepth` or "
        f"`locate`, `command -v` → `rpm -q`). Max 2 tool_calls per turn."
    )
    return {"role": "user", "content": body}


# Reason-targeted corrective nudges. Sent ONCE per turn after a
# recoverable failure (model thought aloud / refused / echoed). Kept
# short on purpose — a long instruction at the END of the transcript
# fights with the system prompt for attention; a sharp one-liner is
# more effective on small models.
_RETRY_NUDGES = {
    "thinking_only": (
        "Your previous reply had `thoughts` but no action. Output the "
        "JSON object NOW with EITHER a non-empty `tool_calls` array "
        "(containing the shell command you said you would run) OR a "
        "concrete `final_answer`. Do not respond with thoughts only."
    ),
    "refusal": (
        "Refusing is not allowed. The user owns this machine and asked "
        "a legitimate sysadmin question. Output the JSON object with "
        "EITHER a tool_call (e.g. `find $HOME -type d -iname '<name>*'`) "
        "OR a concrete `final_answer`. No apologies, no caveats, no "
        "'sensitive issue' language."
    ),
    "echo": (
        "Your previous reply just paraphrased the user's question. "
        "Output the JSON object now with EITHER a real tool_call (a "
        "shell command that produces facts) OR a `final_answer` that "
        "contains NEW information. Do not echo the user's words."
    ),
    "field_leak": (
        "Your previous `final_answer` was a single schema keyword "
        "(like \"purpose\", \"task\", \"command\", or \"shell\") instead of "
        "a real reply. `final_answer` MUST be the actual user-facing "
        "markdown answer: 1-3 sentences of plain prose, optionally "
        "followed by a fenced ```bash``` block. Output the JSON object "
        "now with EITHER a real tool_call OR a real `final_answer`."
    ),
}


def run_agent_turn(
    messages: list[dict],
    *,
    chat_url: str,
    generate_url: str,
    model: str,
    num_ctx: int | None = None,
    _allow_retry: bool = True,
) -> dict:
    """Run one assistant turn. Returns the normalized response dict plus
    the raw model output for debugging.

    If the first response is a recoverable failure (`thinking_only`,
    `refusal`, `echo`), we send ONE corrective follow-up message before
    giving up. This costs an extra round-trip on failure but reliably
    converts "I will search for it." (no action) into the actual
    `find …` tool_call on q4 coder models.
    """
    raw = _ollama_chat_with_fallback(
        messages,
        chat_url=chat_url,
        generate_url=generate_url,
        model=model,
        num_ctx=num_ctx,
    )
    # Pull the most recent user message — the echo detector compares
    # against it. We walk from the end so we pick up the *current* turn's
    # question even when the transcript already has tool-result messages
    # appended after it.
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user = m["content"]
            break
    norm = _normalize_response(raw, user_query=last_user)

    reason = norm.get("_failure_reason")
    if (
        _allow_retry
        and not norm.get("_schema_ok")
        and reason in _RETRY_NUDGES
    ):
        retry_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _RETRY_NUDGES[reason]},
        ]
        try:
            raw2 = _ollama_chat_with_fallback(
                retry_messages,
                chat_url=chat_url,
                generate_url=generate_url,
                model=model,
                num_ctx=num_ctx,
            )
        except Exception:
            raw2 = ""
        if raw2:
            norm2 = _normalize_response(raw2, user_query=last_user)
            # Only adopt the retry if it actually fixed the contract.
            # Otherwise stick with the original — a worse retry would
            # mislead the caller into thinking we made progress.
            if norm2.get("_schema_ok"):
                norm2["raw"] = raw2
                norm2["_retry_reason"] = reason
                return norm2

    norm["raw"] = raw
    return norm


# ---------------------------------------------------------------------------
# Turn summarization (cheap memory for the next turn)
# ---------------------------------------------------------------------------
#
# Why a separate call? Two reasons:
#
# 1. Schema isolation. The main loop's JSON contract is already crowded
#    (thoughts/tool_calls/final_answer). Bolting summary+next_step onto every
#    turn would degrade adherence on a 1.5B model — its output gets noisier as
#    the schema grows. Asking for the recap *after* the answer is locked in
#    keeps each generation tightly scoped.
# 2. Token budget. We only summarize once per user turn (when done=true), so
#    the cost is amortized. The summarizer prompt is intentionally tiny so it
#    stays fast on CPU-only fallback.

_SUMMARY_SYSTEM_PROMPT = (
    "You compress a finished Basher (Fedora bash assistant) exchange into a "
    "tiny memory note for the next turn. Output ONLY this JSON, nothing else:\n\n"
    "{\n"
    '  "summary":   "<=2 sentences. What the user asked + what was answered. No fluff. Empty string if the exchange had no real content.>",\n'
    '  "next_step": "<a single concrete follow-up the user is likely to ask next, derived ONLY from topics actually present in the exchange. Empty string by default.>"\n'
    "}\n\n"
    "Hard rules — violating any of these means output \"\" for that field:\n"
    "1. Write in the user\'s language (Finnish in, Finnish out).\n"
    "2. Never invent facts, commands, packages, or tools that are NOT in the exchange.\n"
    "   In particular: do NOT suggest podman, systemd, dnf, htop, etc. unless the user "
    "   or assistant ALREADY mentioned them.\n"
    "3. `next_step` must be runnable as a fresh user prompt (e.g. \"show kernel version\", "
    "   \"asenna git\"), NOT meta-commentary like \"the user might want to...\".\n"
    "4. If the user just said hello / what\'s up / thanks / a single word with no topic, "
    "   set both `summary` and `next_step` to \"\".\n"
    "5. If the assistant\'s answer was empty, an error, or starts with \"(\", "
    "   set both fields to \"\".\n"
    "6. Keep `summary` under ~240 chars and `next_step` under ~80 chars."
)


def summarize_turn(
    *,
    user_query: str,
    final_answer: str,
    chat_url: str,
    generate_url: str,
    model: str,
    prior_summary: str = "",
    num_ctx: int | None = None,
) -> dict:
    """Compress a finished agent turn into {summary, next_step}.

    Returns a dict with both fields always present (empty string on
    failure). Never raises — summarization is best-effort; if it fails we
    just lose context for the next turn, which is recoverable.
    """
    user_query = (user_query or "").strip()
    final_answer = (final_answer or "").strip()
    if not user_query or not final_answer:
        return {"summary": "", "next_step": ""}

    # Cheap guard: skip the LLM round-trip entirely on inputs that are
    # almost guaranteed to produce hallucinated next_steps.
    #   - synthetic placeholder answers ("(no answer)", "(no response)", ...)
    #   - tiny / smalltalk queries with no topic for the model to latch onto
    if final_answer.startswith("(") and final_answer.endswith(")"):
        return {"summary": "", "next_step": ""}
    _SMALLTALK = {
        "hi", "hei", "moi", "moikka", "terve", "hey", "hello", "yo",
        "thanks", "thx", "kiitos", "kiitti", "sup", "what's up", "whats up",
        "hmm", "ok", "okay", "joo", "ei",
    }
    if user_query.lower().strip("?!. ") in _SMALLTALK:
        return {"summary": "", "next_step": ""}

    prior = (prior_summary or "").strip()
    transcript_parts = []
    if prior:
        transcript_parts.append(f"# Prior summary\n{prior[:600]}")
    transcript_parts.append(f"# User\n{user_query[:1200]}")
    transcript_parts.append(f"# Assistant final answer\n{final_answer[:2000]}")
    transcript = "\n\n".join(transcript_parts)

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    try:
        raw = _ollama_chat_with_fallback(
            messages,
            chat_url=chat_url,
            generate_url=generate_url,
            model=model,
            temperature=0.1,   # deterministic recap
            top_p=0.85,
            num_predict=160,   # plenty for two short fields
            num_ctx=num_ctx,
            timeout=60,
            force_json=True,
        )
    except Exception:
        return {"summary": "", "next_step": ""}

    parsed = _try_parse_json(raw) or {}
    summary = str(parsed.get("summary") or "").strip()
    next_step = str(parsed.get("next_step") or "").strip()
    if len(summary) > 600:
        summary = summary[:600].rstrip() + "..."
    if len(next_step) > 200:
        next_step = next_step[:200].rstrip()

    # Grounding check on next_step. Small models love to echo terms from
    # the system prompt's few-shot examples ("podman", "systemd",
    # "journalctl") even when the exchange had nothing to do with them.
    # If the suggestion mentions a notable token that appears nowhere in
    # the actual exchange, drop it.
    if next_step:
        haystack = (user_query + "\n" + final_answer + "\n" + prior).lower()
        suspect_tokens = (
            "podman", "systemd", "systemctl", "journalctl",
            "selinux", "firewall-cmd", "dnf", "rpm", "flatpak",
            "docker", "kubectl", "git", "ssh",
        )
        leaked = [t for t in suspect_tokens if t in next_step.lower() and t not in haystack]
        if leaked:
            next_step = ""
    return {"summary": summary, "next_step": next_step}


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
    "OllamaError",
    "classify_command",
    "is_banned",
    "build_initial_messages",
    "format_tool_result_message",
    "run_agent_turn",
    "append_assistant_turn",
    "summarize_turn",
    "truncate_for_history",
]
