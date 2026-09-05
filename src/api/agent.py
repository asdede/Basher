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
`destructive`/`shell`/`task`/`response`), NEVER a placeholder identifier
(`file_location`, `path_to_file`, `<filename>`, `your_answer`, ...), and
NEVER the user's question paraphrased back. If you have nothing useful
to say, emit a `tool_calls` instead.

NEVER invent paths, filenames, package names, port numbers, or version
strings. If you don't have a real value from a tool result or a doc, you
MUST emit a `tool_calls` to go get one. Tokens like `/home/user/...`,
`/path/to/...`, `example.com`, `localhost:PORT` are forbidden as
substitutes for real data.

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
3. NEVER refuse and NEVER deflect. The user owns their machine and you
   have a live shell on it. Questions like "where is my X folder?",
   "what's running on port N?", "find file Y", "list my containers" are
   normal sysadmin work. The following kinds of replies are FORBIDDEN:
   - "I'm sorry, I can't help with that", "this is a sensitive issue",
     "as an AI model...".
   - "I don't have access to your system / files / machine" — you DO
     (the `shell` tool). Use it.
   - "You may want to try searching online", "consult the documentation",
     "reach out to the support team / sysadmin", "back up your data
     regularly" — these are deflections, not answers.
   If you don't know the value, emit a `find` / `locate` / `command -v` /
   `ss` / `ps` / `grep -r` one-liner that *would* find it.
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
9. **Match the language of the user's CURRENT (most recent) message.**
   The current user turn wins over any prior context, summaries, or
   few-shot examples. If the latest user message is in English, reply
   in English even when older messages, recap text, or system notes
   were in another language. If the latest user message is in Finnish,
   reply in Finnish even when the few-shot examples above are English.
   Do NOT switch languages mid-answer.
10. **When a previous tool call failed** (exit_code != 0, timed out, or
    returned empty stdout for a `find`/`grep`/`locate`), do NOT repeat
    the same command and do NOT emit a placeholder token like
    `find_folder` / `result` / `path_to_file` as `final_answer`. Either
    (a) emit a *different* `tool_calls` (broader `find`, `locate`,
    different path, different pattern, `2>/dev/null` to silence noise),
    or (b) emit a concrete `final_answer` that honestly says what was
    tried and what was not found — in 1-2 short sentences + a fenced
    ```bash``` block the user can re-run with different inputs. Never
    fabricate a path or filename to fill the gap.
11. When you give `final_answer`, be concise: 1-3 short sentences of
    plain prose explaining the result, then a fenced ```bash``` block of
    the exact commands the user should run themselves. Do NOT echo the
    question back. No filler. No apologies. No schema fragments.
12. **When Documentation is attached below**, treat it as the primary
    source for how-to / syntax / flag / procedure answers. Prefer exact
    commands and flags from those docs over memory. If docs and memory
    conflict, trust the docs. Still use tools for *this machine's* live
    state (files, ports, installed packages, running services). Never
    invent flags, paths, or package names that are not in the docs or
    a tool result.

# Examples (study the tool choices)

User: "What Kernel am I running?"
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

Tool result (find ...): "/home/user/code/inno/Basher\\n/home/user/code/inno/Basher/src/agent-ui/node_modules/.../basher-agent-ui"
Assistant:
{"thoughts":"Found it: parent of the project is the code folder.","tool_calls":[],"final_answer":"Your code folder is **/home/user/code** (the project lives at `/home/masterbates/code/inno/Basher`).\\n\\n```bash\\nls /home/masterbates/code\\n```"}

# Wrong vs right (final_answer content)

User: "what kernel am I on?"
Tool result (uname -r): "6.14.6-300.fc43.x86_64"
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"purpose"}
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"task"}
WRONG  → {"thoughts":"...","tool_calls":[],"final_answer":"what kernel am I on"}
   ↑ One-word / schema-key / question-echo answers are forbidden.
RIGHT  → {"thoughts":"...","tool_calls":[],"final_answer":"Kernel **6.14.6-300.fc43.x86_64**.\\n\\n```bash\\nuname -r\\n```"}

# Wrong vs right (the "I lost saliapp" failure mode)

User: "I have lost folder called saliapp, can you find it for me?"
WRONG  → {"thoughts":"User lost saliapp.","tool_calls":[],"final_answer":"saliapp"}
   ↑ Echoing the keyword from the question is NOT an answer. The user
   already knows the folder is called saliapp — that's why they asked.
WRONG  → {"thoughts":"saliapp lives in /usr/local/bin.","tool_calls":[],"final_answer":"/usr/local/bin/saliapp"}
   ↑ You did not run a tool. Any path you write here is a guess. The
   user's actual folder could be anywhere — `~/code/saliapp`,
   `~/Documents/saliapp`, `/opt/saliapp`, ... You CANNOT know without
   searching.
RIGHT  → {"thoughts":"Search $HOME for any directory named saliapp; do not guess the path.","tool_calls":[{"type":"shell","command":"find \\"$HOME\\" -maxdepth 6 -type d -iname 'saliapp*' 2>/dev/null | head -20","purpose":"locate the saliapp folder","danger":"safe"}],"final_answer":null}

Tool result (find ...): "/home/user/projects/saliapp"
RIGHT  → {"thoughts":"Found it under ~/projects.","tool_calls":[],"final_answer":"Your **saliapp** folder is at `/home/user/projects/saliapp`.\\n\\n```bash\\nls /home/user/projects/saliapp\\n```"}

If the search returns nothing, do NOT invent a path. Answer honestly:
{"thoughts":"Nothing under $HOME; admit it and suggest a broader search.","tool_calls":[],"final_answer":"I could not find any folder named `saliapp*` under `$HOME`. Try a wider search:\\n\\n```bash\\nsudo find / -type d -iname 'saliapp*' 2>/dev/null\\n```"}
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

    # Order matters here. The most actionable retry nudge wins, since
    # `failure_reason = failure_reason or "..."` is short-circuit.
    #
    # 1. refusal       — safety-style language; treat first, irrespective
    #                    of whether a tool ran. The cure (drop the safety
    #                    reflex) is the same either way.
    # 2. no_tool_for_state — state question with no tool_call. This is the
    #                    ROOT CAUSE behind most "saliapp" / "/usr/local/bin/
    #                    saliapp" failures; echo/field_leak are just the
    #                    symptom. Putting it before those two means the
    #                    user gets the more actionable nudge ("emit
    #                    `find …`") instead of the generic "give new info".
    # 3. echo          — paraphrase of the question; rare without #2.
    # 4. field_leak    — schema keyword / placeholder.

    # 1. Refusal detection. Coder models at q4 have residual safety
    # tuning that fires unpredictably ("I'm sorry, I can't assist with
    # that", "this is a sensitive political issue", etc.) for benign
    # sysadmin questions. Letting that through is worse than admitting
    # the turn failed: the user can rephrase but a canned refusal wastes
    # their time and poisons the rolling memory.
    if isinstance(final_answer, str) and _is_refusal(final_answer):
        schema_ok = False
        failure_reason = failure_reason or "refusal"

    # 2. No-tool-for-state-question detection. Purely structural: the
    # user asked a state-on-this-machine question, and the model
    # committed a `final_answer` without running anything. That's
    # always wrong — the answer is at best a guess, more usually a
    # hallucinated path (`/usr/local/bin/saliapp`) or a one-word noun
    # echo (`saliapp`). The retry nudge forces an actual `find` /
    # `locate` tool call before any `final_answer`. We DO allow
    # tool-result callbacks: those messages start with "[tool_result"
    # and have already been preceded by a real tool, so this turn is
    # the post-tool reasoning, not the guess-the-answer turn.
    if (
        isinstance(final_answer, str)
        and not tool_calls
        and not (user_query or "").startswith("[tool_result")
        and _is_state_question(user_query)
    ):
        schema_ok = False
        failure_reason = failure_reason or "no_tool_for_state"

    # 3. Echo detection. Models occasionally satisfy `format=json` by
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

    # 4. Field-leak detection. A small model under format=json sometimes
    # fills `final_answer` with one of the schema keys ("purpose",
    # "task", "command", ...) — JSON-valid, completely useless. Catch
    # it so the retry-nudge loop can re-prompt for a real answer
    # instead of the renderer showing "purpose" as if it were a reply.
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


# Short-response refusals — the classic "I'm sorry, I can't ..." opener.
# Capped tight so a long real answer that happens to contain a matching
# substring doesn't get tripped.
_REFUSAL_PATTERNS = (
    re.compile(r"i\s*[''']?m\s+sorry,?\s+(but\s+)?i\s+(can(?:not|'?\s*t)|am\s+unable|won'?t)", re.I),
    re.compile(r"i\s+(?:can(?:not|'?\s*t)|am\s+unable|won'?t)\s+(?:assist|help|answer|address|comment)", re.I),
    re.compile(r"as\s+an\s+ai(?:\s+language)?\s+model,?\s+i", re.I),
    re.compile(r"i'?m\s+not\s+able\s+to\s+(?:provide|assist|help|answer|address)", re.I),
    re.compile(r"sensitive\s+(?:and\s+)?(?:political|legal|religious)\s+issue", re.I),
    re.compile(r"won'?t\s+comment\s+on\s+(?:or\s+express\s+opinions\s+)?(?:about\s+)?(?:political|legal|religious)", re.I),
    re.compile(r"i\s+don'?t\s+(?:engage|comment)\s+(?:with|on)\s+(?:political|controversial)", re.I),
)

# Long-response refusals — the model writes 3-5 sentences of "I'm sorry
# to hear ... I don't have access to your system ... reach out to the
# support team / search online / back up your data". Looks polite, is
# pure refusal-by-deflection for a sysadmin question we explicitly told
# the model it CAN answer with a tool call. We accept it only if at
# least TWO of these tells fire (single occurrences happen in legit
# explanations).
_LONG_REFUSAL_TELLS = (
    re.compile(r"i\s+don'?t\s+have\s+(?:access\s+to|the\s+ability\s+to\s+access)\s+(?:your|the\s+user'?s)\s+(?:system|machine|files|filesystem|computer)", re.I),
    re.compile(r"i'?m\s+(?:not\s+able|unable)\s+to\s+(?:provide|access|see|read|browse|search|look\s+up)", re.I),
    re.compile(r"(?:please\s+)?(?:reach\s+out|contact|consult)\s+(?:the\s+)?(?:support\s+team|administrator|sysadmin|community|documentation)", re.I),
    re.compile(r"(?:you\s+may\s+want\s+to|you\s+could|consider)\s+(?:try(?:ing)?\s+)?(?:searching|googling|looking\s+(?:up|online))", re.I),
    re.compile(r"(?:as\s+an\s+ai|i'?m\s+(?:just\s+)?an\s+ai|i\s+do\s+not\s+have\s+(?:personal|real-time|live))", re.I),
    re.compile(r"(?:keep\s+track\s+of|back(?:\s+|-)up)\s+(?:your\s+)?(?:important\s+)?(?:files|data|paths)\s+(?:regularly|to\s+prevent)", re.I),
)


def _is_refusal(text: str) -> bool:
    """Heuristic: does this look like a canned safety refusal? Two layers:
    short stock refusals (≤320 chars) and long "I can't access your
    system, please contact support" deflections (any length, needs ≥2
    independent tells to fire)."""
    s = (text or "").strip()
    if not s:
        return False
    if len(s) <= 320 and any(p.search(s) for p in _REFUSAL_PATTERNS):
        return True
    # Long deflection: count distinct refusal tells. Two or more = canned.
    hits = sum(1 for p in _LONG_REFUSAL_TELLS if p.search(s))
    return hits >= 2


# Tokens the model knows from the system prompt: schema keys, schema
# enum values, tool type names, plus the generic placeholder words
# (`result`, `value`, `path`, `location`, ...) that small models like
# to emit when they can't follow through. Under `format: "json"` a 1.5b
# coder sometimes "satisfies" the contract by filling `final_answer`
# with one of these verbatim. The result parses as valid JSON but tells
# the user nothing — surfacing it is worse than admitting the turn
# failed, because the rolling memory then caches "purpose" or "task"
# as if it were a real reply.
_SCHEMA_LEAK_TOKENS = frozenset({
    "thoughts", "tool_calls", "tool_call", "final_answer", "command",
    "purpose", "danger", "safe", "caution", "destructive", "shell",
    "task", "tasks", "response", "answer", "text", "content", "output",
    "message", "reply", "null", "none", "todo", "tbd",
    # Generic placeholder words. These are never a useful one-word
    # reply in Basher — a real path starts with "/", a real value
    # contains digits, etc.
    "result", "results", "value", "data", "info", "details",
    "placeholder", "example", "filename", "path", "pathname",
    "location", "file_location", "file_path", "filepath",
    "your_answer", "the_answer", "here",
})

# Snake_case identifier like `file_location`, `path_to_file`,
# `your_file_path`. Real one-word answers are almost never shaped this
# way; placeholder fills almost always are.
_SNAKE_PLACEHOLDER_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

# Angle-bracketed template like `<filename>`, `<path>`, `<your answer>`.
_ANGLE_PLACEHOLDER_RE = re.compile(r"^<[^<>]{1,40}>$")


def _looks_like_field_leak(answer: str) -> bool:
    """Detect when `final_answer` is a schema keyword, a snake_case
    placeholder identifier, or an angle-bracketed template instead of an
    actual user-facing reply.

    Concrete failure modes seen in the wild on q4 coder models:
      final_answer: "purpose"
      final_answer: "task"
      final_answer: "shell"
      final_answer: "command"
      final_answer: "file_location"
      final_answer: "path_to_file"
      final_answer: "<filename>"
    All technically valid JSON strings, all useless.
    """
    s = (answer or "").strip().strip("`'\"")
    if not s:
        return False
    # Only consider short single-token answers — anything with whitespace
    # or newlines is at worst an echo, handled separately.
    if " " in s or "\n" in s:
        return False
    if len(s) > 40:
        return False
    if s.lower() in _SCHEMA_LEAK_TOKENS:
        return True
    if _ANGLE_PLACEHOLDER_RE.match(s):
        return True
    if _SNAKE_PLACEHOLDER_RE.match(s.lower()):
        return True
    return False


# Server-side substitution for `final_answer` values that survive the
# retry loop. The renderer used to forward strings like "find_folder"
# verbatim through its "model declined or returned an unusable response:"
# path — technically honest, but the user just sees a meaningless token
# and assumes the agent is broken (cf. "Now it just gives me 'find_folder'
# output when I ask it to find folder called 'saliapp'"). Replacing the
# leaked token with an actionable English explanation keeps the user in
# the loop without polluting the rolling memory: the renderer already
# refuses to summarize / cache turns where `schema_ok` is false, so this
# substitution never feeds back into future prompts.
_FAILURE_EXPLANATIONS = {
    "field_leak": (
        "The model returned a placeholder token instead of a real reply "
        "(the previous tool call most likely failed or returned nothing, "
        "and the corrective retry could not recover). Try rephrasing your "
        "question with more specifics, or press F2 and run the search "
        "yourself."
    ),
    "echo": (
        "The model only paraphrased your question instead of answering. "
        "Rephrase with more specifics, or press F2 and run the command "
        "yourself."
    ),
    "refusal": (
        "The model refused to answer. Rephrase the question, or press F2 "
        "and run the command yourself."
    ),
    "thinking_only": (
        "The model planned a tool call but never followed through. Try "
        "the question again, or press F2 and run the command yourself."
    ),
    "empty": (
        "The model returned an empty response. Try rephrasing the question, "
        "or press F2 and run the command yourself."
    ),
    "non_json": (
        "The model returned malformed JSON we could not parse. Try "
        "rephrasing the question, or press F2 and run the command yourself."
    ),
    "no_tool_for_state": (
        "The model tried to answer a question about your machine without "
        "actually running a shell tool — any path or status it wrote "
        "would have been a guess (the common failure mode is echoing "
        "the keyword from the question, or fabricating a path like "
        "`/usr/local/bin/<name>`). The corrective retry did not produce "
        "a valid `tool_calls` either. Try rephrasing more concretely "
        "(e.g. \"find folder named X under $HOME\") or press F2 and run "
        "`find \"$HOME\" -iname 'NAME*'` yourself."
    ),
}


def _sanitize_failed_final_answer(norm: dict) -> None:
    """Mutate `norm` in place so a post-retry failure carries a clear
    English explanation instead of the raw garbage the model emitted.

    The original text is preserved in `_raw_final_answer` for debug logs.
    Only fires when `_schema_ok` is already False — successful turns are
    never touched."""
    if norm.get("_schema_ok"):
        return
    reason = norm.get("_failure_reason") or ""
    raw_fa = str(norm.get("final_answer") or "").strip()
    explanation = _FAILURE_EXPLANATIONS.get(reason)
    if not explanation:
        return
    # Preserve the model's actual words so we can debug from logs / the
    # debug=True API payload, but only when they were short/parseable
    # enough to be useful. Long refusal essays are not helpful here.
    norm["_raw_final_answer"] = raw_fa[:240]
    # Append the leaked token (if short enough) so an operator skimming
    # the UI can still see exactly what the model said, in a clean form.
    if raw_fa and len(raw_fa) <= 80 and reason == "field_leak":
        norm["final_answer"] = (
            f"{explanation}\n\n_Model emitted:_ `{raw_fa}`"
        )
    else:
        norm["final_answer"] = explanation


def _word_tokens(s: str) -> list[str]:
    """Split a string into word tokens, stripping ALL punctuation —
    including commas, periods, parens, quotes. The earlier
    `re.split(r"\\s+", s)` version left `saliapp,` as one token, which
    let single-noun echoes like ("I have lost folder called saliapp,
    can you find it?" -> "saliapp") slip past `_looks_like_echo`
    because `saliapp != saliapp,`."""
    return [t for t in re.split(r"\W+", (s or "").lower()) if t]


def _looks_like_echo(query: str, answer: str) -> bool:
    """Detect when the model lazily paraphrased the question into the
    answer instead of running a tool or giving real info.

    Concrete failure modes seen in the wild on q4 coder models:
      user: "Can you find it with that information?"
      answer: "find it with that information"
      user: "I have lost folder called saliapp, can you find it?"
      answer: "saliapp"
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
    a_tokens = _word_tokens(a)
    if not a_tokens:
        return False
    q_tokens = set(_word_tokens(q))
    overlap = sum(1 for t in a_tokens if t in q_tokens) / len(a_tokens)
    has_signal = any(
        ("/" in t) or t.startswith("$") or t.startswith("`") or
        any(c.isdigit() for c in t)
        for t in a_tokens
    )
    return overlap >= 0.85 and not has_signal


# -----------------------------------------------------------------------
# State-question detector
# -----------------------------------------------------------------------
# A "state question" is one whose accurate answer is on the user's
# machine RIGHT NOW (file location, port status, running process,
# installed package, etc.) — not anything the model can answer from
# pre-training. For these, emitting a `final_answer` without first
# running a `tool_calls` is always wrong, regardless of how plausible
# the answer looks. This is the failure mode behind the entire
# "saliapp" / "/usr/local/bin/saliapp" hallucination chain: the model
# never actually searched, it just guessed.
#
# We cover English and Finnish because the project is Finnish-authored
# and the user routinely mixes both. A query that matches ANY of these
# patterns AND comes back with `tool_calls == []` is flagged as
# `no_tool_for_state` and gets ONE forced corrective retry that
# explicitly demands a shell command.
_STATE_QUESTION_PATTERNS = (
    # English: find / locate / lost / search intent for a file or folder.
    re.compile(r"\bfind\s+(?:me\s+)?(?:the\s+|my\s+|a\s+|an\s+|it|them|this|that|some\b)", re.I),
    re.compile(r"\bfind\s+\w+\s*(?:folder|file|directory|dir)\b", re.I),
    re.compile(r"\bfind\s+(?:folder|file|directory|dir|path)\s+(?:called|named)\b", re.I),
    re.compile(r"\b(?:locate|search\s+for)\b", re.I),
    re.compile(r"\b(?:lost|misplaced)\b", re.I),
    re.compile(r"\bi\s+(?:have\s+)?(?:lost|misplaced)\b", re.I),
    re.compile(r"\bi'?ve?\s+(?:lost|misplaced)\b", re.I),
    re.compile(r"\bcan(?:not|'?\s*t)\s+find\b", re.I),
    # English: "where is / where can I find / where's" intent.
    re.compile(r"\bwhere\s+(?:is|are|can\s+i\s+(?:find|see))\b", re.I),
    re.compile(r"\bwhere'?s\b", re.I),
    # English: "what(')s the path/location/version/kernel/..." — answer
    # is on this machine right now, not in pre-training. `what's`,
    # `whats`, `what is` all accepted (small models drop apostrophes).
    re.compile(r"\bwhat(?:'?s|\s+is)?\s+(?:the\s+)?(?:path|location|directory|folder)\s+(?:for|to|of)\b", re.I),
    re.compile(r"\bpath\s+(?:for|to|of)\s+(?:the\s+|my\s+)?\w", re.I),
    re.compile(r"\bwhat(?:'?s|\s+is)?\s+(?:my\s+)?(?:kernel|os|distro|release|hostname|user|shell|uptime|ip|cpu|memory|ram)\b", re.I),
    re.compile(r"\bwhat\s+(?:version\s+of\s+\w+|kernel|os|distro)\s+(?:am\s+i|do\s+i|is\s+(?:my|on))\b", re.I),
    re.compile(r"\bam\s+i\s+(?:on|using|running)\s+\w", re.I),
    re.compile(r"\bwhich\s+(?:version|kernel|distro)\b", re.I),
    # English: state questions about running processes / ports / packages.
    re.compile(r"\bwhat(?:'?s|\s+is)?\s+(?:on|running|listening|using)\s+(?:port|pid)\b", re.I),
    re.compile(r"\bis\s+\w+\s+(?:installed|running|enabled|active|loaded)\b", re.I),
    re.compile(r"\bdo\s+i\s+have\s+(?:\w+\s+)?(?:installed|running)\b", re.I),
    re.compile(r"\bdo\s+i\s+have\s+\w+\s*\?", re.I),
    re.compile(r"\b(?:list|show\s+me)\s+(?:my\s+)?(?:containers|processes|ports|services|units|disks|mounts)\b", re.I),
    # Finnish: missä on / mistä löytyy / kadottanut / hukassa.
    re.compile(r"\bmiss[äa]\s+(?:on|sijaitsee|l[öo]ytyy)\b", re.I),
    re.compile(r"\bmist[äa]\s+l[öo]yty\w*\b", re.I),
    re.compile(r"\bkadottan(?:ut|i)\b", re.I),
    re.compile(r"\bhukass(?:a|sa)\b", re.I),
    re.compile(r"\b(?:l[öo]yd[äa]\w*|etsi\w*)\s+(?:kansio|tiedosto|polku)", re.I),
    re.compile(r"\bonko\s+\w+\s+(?:asennett\w+|k[äa]ynniss[äa]|aktiivin\w+)\b", re.I),
    re.compile(r"\bmik[äa]\s+on\s+(?:polku|kansio|tiedosto)\b", re.I),
)

# Phrases that LOOK stateful but are actually general-knowledge questions
# the model should answer from training. We bypass the state-question
# check when ANY of these match — e.g. "how do I find process IDs" is a
# how-to, not "find a specific file for me".
_STATE_QUESTION_BLOCKLIST = (
    re.compile(r"\bhow\s+(?:do|can|to)\b", re.I),
    re.compile(r"\bwhat\s+(?:is|does|are)\s+(?:the\s+)?(?:difference|command|meaning|purpose)\b", re.I),
    re.compile(r"\b(?:tutorial|guide|example|documentation)\b", re.I),
    re.compile(r"\bmit[äa]\s+(?:tarkoitt\w+|on\s+\w+\s+komento)\b", re.I),
)


def _is_state_question(query: str) -> bool:
    """True when the query needs a fresh shell tool call to answer
    accurately. We use this to refuse first-turn `final_answer`s that
    skip the tool — see `no_tool_for_state` in `_normalize_response`."""
    q = (query or "").strip()
    if not q or len(q) > 600:
        # Very long messages are usually multi-paragraph explanations
        # or pasted logs; don't try to classify them with regex.
        return False
    if any(p.search(q) for p in _STATE_QUESTION_BLOCKLIST):
        return False
    return any(p.search(q) for p in _STATE_QUESTION_PATTERNS)


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
# RAG routing — when the agent should pull documentation
# ---------------------------------------------------------------------------

# Live host state → tools only (docs would dilute a small model).
_TOOL_FIRST_RE = re.compile(
    r"(?is)\b("
    r"where\s+(is|are|can\s+i\s+find)|find\s+(my|the)\s+(file|folder|dir|directory)|"
    r"lost\s+my|what('s|\s+is)\s+(on|using)\s+(my\s+)?port|"
    r"(is|are)\s+.{0,60}(running|installed|active|enabled)\b|"
    r"list\s+(my|the)\s+(files|containers|processes|packages|services)|"
    r"disk\s+(space|usage)|free\s+space|my\s+(home|repo|branch|hostname)|"
    r"on\s+(this|my)\s+(machine|system|host|computer)|"
    r"what\s+process|which\s+(file|port|process)|"
    r"show\s+me\s+(my|what)|do\s+i\s+have\s+\w+\s+installed|"
    r"what\s+kernel\s+am\s+i|am\s+i\s+root"
    r")\b"
)

# How-to / explain / syntax → retrieve docs.
_DOC_FIRST_RE = re.compile(
    r"(?is)\b("
    r"how\s+(do|to|can|should|would)|what\s+(is|are|does|do)|explain|"
    r"difference\s+between|versus|\bvs\.?\b|syntax|meaning\s+of|"
    r"what\s+(flag|option|parameter|switch)|best\s+practice|"
    r"why\s+(does|do|is|are|should)|write\s+a\s+(bash\s+|shell\s+)?script|"
    r"example\s+of|document(ation)?\s+(for|about)|"
    r"selinux|dnf-automatic|systemd\s+unit|dockerfile|"
    r"quote|quoting|array|subshell|pipeline|redirection|"
    r"set\s+-u|trap\s+|chmod|chown"
    r")\b"
)


def should_use_retrieval(query: str) -> bool:
    """
    Decide whether agent mode should inject RAG documentation.

    Tool-first for live machine state; docs for how-to / explain / syntax.
    Conservative default: False when ambiguous (keeps the tool loop sharp).
    """
    q = (query or "").strip()
    if not q:
        return False
    if _TOOL_FIRST_RE.search(q):
        return False
    if _DOC_FIRST_RE.search(q):
        return True
    return False


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
            "GROUNDING RULES:\n"
            "- Prefer facts, flags, and procedures from these docs over memory.\n"
            "- When answering how-to questions, use the exact commands/flags "
            "shown below when they fit.\n"
            "- If docs and memory conflict, trust the docs for Fedora / bash / "
            "git / docker.\n"
            "- Do NOT invent flags, paths, or package names absent from the "
            "docs or a later tool result.\n"
            "- Docs do not know THIS host — still use tools for live state.\n\n"
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

    Note on guidance text: the model often emits a `find_folder` /
    `path_to_file` placeholder when it sees `exit_code: 1` + empty stdout
    (the tool errored or returned nothing). We give it a per-outcome
    decision tree right after the result so it has a concrete next move
    instead of pattern-matching a JSON token. Small models do MUCH
    better with "if X then Y" cheat-sheets than with abstract advice.
    """
    stdout_raw = str(result.get("stdout", "") or "")
    stderr_raw = str(result.get("stderr", "") or "")
    stdout = truncate_for_history(stdout_raw)
    stderr = truncate_for_history(stderr_raw)
    exit_code = result.get("exit_code", result.get("code", "unknown"))
    timed_out = bool(result.get("timed_out", False))

    # Tailor the post-result coaching to what actually happened. A small
    # model can latch onto the wrong reaction (e.g. emit `find_folder`
    # placeholder when a find returned nothing) — being explicit here
    # avoids that whole class of failure.
    try:
        ec_int = int(exit_code) if exit_code not in ("unknown", None) else None
    except (TypeError, ValueError):
        ec_int = None
    stdout_empty = stdout_raw.strip() == ""
    stderr_empty = stderr_raw.strip() == ""

    if timed_out:
        guidance = (
            "The command TIMED OUT. Do NOT repeat it as-is. Either narrow "
            "the search (smaller `-maxdepth`, more specific pattern, "
            "scope to a directory), or emit a `final_answer` that tells "
            "the user the search timed out and suggests a narrower command "
            "for them to run."
        )
    elif ec_int is not None and ec_int != 0:
        guidance = (
            f"The command FAILED with exit_code={ec_int}. Read the stderr "
            "above to understand why. Do NOT repeat the same command and "
            "do NOT emit a placeholder token like `find_folder` / `result` "
            "/ `path_to_file` as `final_answer`. Pick exactly ONE next "
            "action:\n"
            "  (a) If you can fix the command (typo, wrong path, missing "
            "flag, needs `2>/dev/null`), emit a corrected `tool_calls`.\n"
            "  (b) If the command itself is wrong for the question, "
            "SWITCH tools (e.g. `ls` -> `find -iname`, `find` -> `locate`, "
            "`command -v` -> `rpm -q`).\n"
            "  (c) If you have enough info to answer despite the error, "
            "emit a real `final_answer` (1-2 sentences + fenced ```bash``` "
            "block) that honestly says what was tried, what failed, and "
            "what the user can run themselves."
        )
    elif stdout_empty and stderr_empty:
        guidance = (
            "The command SUCCEEDED but produced NO output — for a search "
            "(`find`/`grep`/`locate`/`command -v`) that almost always means "
            "nothing was found. Do NOT repeat the same command and do NOT "
            "emit a placeholder token like `find_folder` as `final_answer`. "
            "Pick ONE: (a) broaden the search (larger `-maxdepth`, drop "
            "`-type`, broader pattern, search a different root like `/`, "
            "`/etc`, `/var`, `/opt`), or (b) emit a real `final_answer` "
            "that says clearly that nothing matched and shows the exact "
            "command in a fenced ```bash``` block so the user can refine it."
        )
    else:
        guidance = (
            "Reason about the result and reply with the JSON object only. "
            "If the result answers the question, output `final_answer` now "
            "(real markdown sentence + fenced code block, NEVER a single "
            "schema keyword or placeholder token like `find_folder`). If "
            "the result is partial or noisy, you may emit ONE more "
            "`tool_calls` to narrow it down."
        )

    body = (
        f"[tool_result id={tool_call_id} command={json.dumps(command)}]\n"
        f"exit_code: {exit_code}\n"
        f"timed_out: {timed_out}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}\n"
        f"[/tool_result]\n\n"
        f"{guidance}\n"
        f"Max 2 tool_calls per turn. Match the language of the original "
        f"user question (not this system note)."
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
        "Your previous `final_answer` was a placeholder token (a schema "
        "keyword like \"purpose\"/\"command\"/\"shell\", a snake_case "
        "identifier like \"file_location\"/\"path_to_file\", or an angle "
        "template like \"<filename>\") instead of a real reply. "
        "`final_answer` MUST be the actual user-facing markdown answer: "
        "1-3 sentences of plain prose, optionally followed by a fenced "
        "```bash``` block, containing REAL values from the conversation "
        "(never invented paths like \"/home/user/...\"). If you do not "
        "have the real value yet, output a `tool_calls` (e.g. "
        "`find \"$HOME\" -iname '...'`) instead. Output the JSON object now."
    ),
    "no_tool_for_state": (
        "Your previous reply was a `final_answer` to a question that "
        "requires looking at the user's machine right now (file/folder "
        "location, running process, port, installed package, etc.). You "
        "did NOT run any tool, so any path, name, or status you wrote "
        "is a guess — that is forbidden. Emit a `tool_calls` NOW with "
        "a concrete shell one-liner (e.g. "
        "`find \"$HOME\" -maxdepth 6 -iname 'saliapp*' 2>/dev/null | head -20` "
        "for \"find folder X\", `ss -tulpn | grep -E ':PORT\\b'` for "
        "\"what's on port N\", `rpm -q PACKAGE` for \"is X installed\"). "
        "`final_answer` is forbidden until you have a real tool result. "
        "Do NOT echo the keyword from the question (like writing "
        "\"saliapp\" by itself) and do NOT fabricate a path "
        "(like \"/usr/local/bin/saliapp\")."
    ),
}


# Forced second-pass reflection. Fires only when the model wants to commit
# a `final_answer` *without* having run any tool — i.e. the whole turn was
# one thinking pass. Policy: never let a single pass commit. We require
# the model to look at its own draft, sanity-check the facts, and either
#   (a) re-emit the same answer (now with explicit verification thoughts), or
#   (b) switch to a tool_call when reflection reveals it's guessing.
# Tool-mediated turns already get >= 2 thinking passes for free (initial
# plan -> tool result -> post-tool reasoning), so we skip reflection there.
MIN_THINKING_PASSES = 2

_REFLECTION_PROMPT = (
    "Reflect once more on your draft answer above before committing. "
    "This is the SECOND thinking pass — required before any final answer.\n\n"
    "Verify, in order:\n"
    "  1. Are every path, filename, package, port, version, and command in "
    "the draft GROUNDED in the actual conversation (a tool result or the "
    "user's own message)? If anything is invented or guessed, you MUST "
    "switch to a `tool_calls` to look it up.\n"
    "  2. Did you actually answer the user's question, or did you drift "
    "(echo it back, suggest searching online, deflect)? If so, fix it.\n"
    "  3. Is the answer in the user's language?\n\n"
    "Now output the JSON object ONCE MORE with:\n"
    "  - `thoughts`: a short note about what you re-checked this pass "
    "(NOT a copy of the previous thoughts).\n"
    "  - EITHER a refined / re-emitted `final_answer` (real markdown, "
    "1-3 sentences + optional fenced ```bash``` block), OR a `tool_calls` "
    "if reflection revealed a missing fact.\n"
    "Never both. Never neither."
)


def run_agent_turn(
    messages: list[dict],
    *,
    chat_url: str,
    generate_url: str,
    model: str,
    num_ctx: int | None = None,
    _allow_retry: bool = True,
    _allow_reflection: bool = True,
) -> dict:
    """Run one assistant turn. Returns the normalized response dict plus
    the raw model output for debugging.

    Two extra round-trips can happen here, both gated and bounded:

    1. **Retry on schema failure.** If the first response is a
       recoverable failure (`thinking_only`, `refusal`, `echo`,
       `field_leak`), we send ONE corrective follow-up. This converts
       "I will search for it." (no action) into the actual `find ...`
       tool_call on q4 coder models.

    2. **Forced reflection (>= 2 thinking passes).** If the (possibly
       retried) response is ready to commit a `final_answer` *without*
       having run any tool — i.e. the whole turn would be a single
       thinking pass — we send ONE reflection prompt asking the model
       to verify its draft. The reflected response then either re-emits
       a refined `final_answer` or escalates to a `tool_calls` (which
       the client will gate behind a permission prompt as usual).
       Tool-mediated turns already get >= 2 thinking passes for free,
       so they skip reflection.

    The returned dict carries `thinking_steps` (a list of thought-strings,
    one per pass) so the UI can show the chain of reasoning, plus
    `_retry_reason` when a corrective nudge was needed.
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

    retry_reason: str | None = None
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
                norm = norm2
                raw = raw2
                retry_reason = reason

    # ------------------------------------------------------------------
    # Forced reflection pass (the "think at least twice" policy).
    # ------------------------------------------------------------------
    # Trigger only when the model is about to commit a `final_answer`
    # straight from a single thinking pass (no tool was used). If we
    # already have tool_calls queued, the caller will execute them and
    # come back with a tool_result message — the post-tool turn IS the
    # second thinking pass, so reflection there would just be wasted
    # round-trips.
    thinking_steps: list[str] = []
    if (
        _allow_reflection
        and norm.get("_schema_ok")
        and norm.get("final_answer")
        and not norm.get("tool_calls")
        and MIN_THINKING_PASSES >= 2
    ):
        first_thoughts = (norm.get("thoughts") or "").strip()

        reflection_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REFLECTION_PROMPT},
        ]
        try:
            raw_r = _ollama_chat_with_fallback(
                reflection_messages,
                chat_url=chat_url,
                generate_url=generate_url,
                model=model,
                num_ctx=num_ctx,
            )
        except Exception:
            raw_r = ""

        adopted_reflection = False
        if raw_r:
            norm_r = _normalize_response(raw_r, user_query=last_user)
            # Only adopt the reflected response if it itself passes the
            # contract — a malformed second pass shouldn't downgrade a
            # clean first pass.
            if norm_r.get("_schema_ok"):
                second_thoughts = (norm_r.get("thoughts") or "").strip()
                if first_thoughts:
                    thinking_steps.append(first_thoughts)
                if second_thoughts:
                    thinking_steps.append(second_thoughts)
                norm = norm_r
                raw = raw_r
                adopted_reflection = True

        if not adopted_reflection and first_thoughts:
            # Reflection failed (network, schema-broken second pass). We
            # still expose the single thinking pass we have so the UI
            # can show it; the caller's "schema_ok" remains as-is.
            thinking_steps.append(first_thoughts)

    if thinking_steps:
        norm["thinking_steps"] = thinking_steps
    if retry_reason and not norm.get("_retry_reason"):
        norm["_retry_reason"] = retry_reason
    # Last-mile substitution: if everything above failed and `final_answer`
    # is still a placeholder token / refusal essay / paraphrased echo,
    # replace it with a clear English explanation so the renderer can show
    # the user something actionable instead of `find_folder`.
    _sanitize_failed_final_answer(norm)
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
    "1. ALWAYS write `summary` in English. It is internal memory for the "
    "   next turn's model, NOT shown to the user. Mixed-language summaries "
    "   bias the next turn into switching languages mid-conversation.\n"
    "2. `next_step` MUST be in the same language as the user's MOST RECENT "
    "   message (the question they actually asked). It is rendered as a "
    "   hint in the terminal — the user sees this string.\n"
    "3. Never invent facts, commands, packages, or tools that are NOT in "
    "   the exchange. In particular: do NOT suggest podman, systemd, dnf, "
    "   htop, etc. unless the user or assistant ALREADY mentioned them.\n"
    "4. `next_step` must be runnable as a fresh user prompt (e.g. "
    "   \"show kernel version\", \"asenna git\"), NOT meta-commentary like "
    "   \"the user might want to...\".\n"
    "5. If the user just said hello / what's up / thanks / a single word "
    "   with no topic, set both `summary` and `next_step` to \"\".\n"
    "6. If the assistant's answer was empty, an error, starts with \"(\", "
    "   or starts with \"The model\" (server failure explanation), set "
    "   both fields to \"\".\n"
    "7. Keep `summary` under ~240 chars and `next_step` under ~80 chars."
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
    #   - server-side failure explanations (sanitized field_leak / refusal
    #     / echo strings — these start with "The model ..." and are
    #     emitted by `_sanitize_failed_final_answer` when the renderer
    #     should NOT cache the turn; summarizing them poisons memory
    #     with "the model could not answer" recap text).
    #   - tiny / smalltalk queries with no topic for the model to latch onto
    if final_answer.startswith("(") and final_answer.endswith(")"):
        return {"summary": "", "next_step": ""}
    _SERVER_FAILURE_PREFIXES = (
        "The model returned a placeholder",
        "The model only paraphrased",
        "The model refused",
        "The model planned a tool call",
        "The model returned an empty response",
        "The model returned malformed JSON",
    )
    if any(final_answer.startswith(p) for p in _SERVER_FAILURE_PREFIXES):
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
    "should_use_retrieval",
    "build_initial_messages",
    "format_tool_result_message",
    "run_agent_turn",
    "append_assistant_turn",
    "summarize_turn",
    "truncate_for_history",
]
