#!/bin/bash
#
# Basher CLI — Ask the Fedora bash assistant.
#
# SYNOPSIS
#   basher.sh [OPTIONS] [QUESTION...]
#   basher.sh [OPTIONS]
#   echo "QUESTION" | basher.sh [OPTIONS]
#
# DESCRIPTION
#   Sends your question to the Basher API and prints the assistant's response.
#   You can pass the question as arguments (no quotes needed) or via stdin.
#
# OPTIONS
#   --help    Show this help and exit.
#   --start   Starts the basher containers (exit 0 on success).
#   --stop    Stops the basher containers (exit 0 on success).
#   --down    Removes the basher containers (volumes kept).
#   --status  Show whether the API is running and healthy.
#   --agent   One-shot /agent/run only. Does NOT execute tools. If the model
#             proposes shell commands, they are printed and the CLI exits 3.
#             Use the Electron desktop UI for approve/run loops.
#
# ENVIRONMENT
#   BASHER_API_URL            Base URL of the API (default: http://localhost:8000).
#   BASHER_DEBUG              If set, show context and retrieval info (e.g. BASHER_DEBUG=1).
#   BASHER_USE_GLOW           If set and glow is installed, render markdown with glow.
#   BASHER_CONNECT_TIMEOUT    Seconds to wait for connection (default: 5).
#   BASHER_HEALTH_MAX_TIME    Max seconds for health check (default: 10).
#   BASHER_ASK_MAX_TIME       Max seconds for /ask request (default: 300).
#   BASHER_USE_GPU            If 1, --start loads docker-compose.gpu.yml (compose Ollama).
#   BASHER_USE_HOST_OLLAMA    If 1, --start uses host Ollama (127.0.0.1:11434) via
#                             docker-compose.host-ollama.yml; does not start compose LLM.
#                             Takes precedence over BASHER_USE_GPU.
#   BASHER_INSTALL_DIR        Repo root (default: parent of Scripts/).
#
# EXAMPLES
#   basher.sh how do I list files
#   basher.sh "one quoted question"
#   echo "how do I use grep?" | basher.sh
#   BASHER_DEBUG=1 basher.sh explain bash arrays
#   BASHER_USE_GPU=1 basher.sh --start
#   BASHER_USE_HOST_OLLAMA=1 basher.sh --start
#
# EXIT CODES
#   0   Success (including --agent with a final_answer and no tools).
#   1   Usage error, API error, compose failure, no response, or invalid JSON.
#   2   API unreachable or not healthy; Docker unavailable for --start/--stop.
#   3   --agent proposed tool calls; CLI does not execute them (use Electron UI).
#   130 Script interrupted (e.g. Ctrl+C).
#

set -e
API_URL="${BASHER_API_URL:-http://localhost:8000}"
# Timeouts so we don't hang when API is down (connect: 5s, health: 10s, ask: 5 min)
BASHER_CONNECT_TIMEOUT="${BASHER_CONNECT_TIMEOUT:-5}"
BASHER_HEALTH_MAX_TIME="${BASHER_HEALTH_MAX_TIME:-10}"
BASHER_ASK_MAX_TIME="${BASHER_ASK_MAX_TIME:-300}"

# Repo root = parent of Scripts/ (where docker-compose.yml lives). Resolve so it works when basher is a symlink or alias.
SCRIPT_PATH="${BASH_SOURCE[0]}"
[[ -L "$SCRIPT_PATH" ]] && SCRIPT_PATH=$(readlink -f "$SCRIPT_PATH" 2>/dev/null || true)
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
BASHER_INSTALL_DIR="${BASHER_INSTALL_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

show_help() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
  echo "Usage: basher.sh [OPTIONS] your question here"
  echo "   or: echo \"Your question\" | basher.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --help   Show this help."
  echo "  --start  Start containers (exit 0 on success)."
  echo "  --stop   Stop containers (exit 0 on success)."
  echo "  --down   Remove containers; volumes kept."
  echo "  --status Show API status."
  echo "  --agent  One-shot agent (exit 3 if tools proposed; use Electron to run)."
  exit 0
}

# --- Lifecycle: compose helpers ---

compose_files() {
  # Prints -f args for docker compose.
  # Host Ollama overlay wins over GPU overlay (different Ollama targets).
  if [[ "${BASHER_USE_HOST_OLLAMA:-0}" == "1" ]]; then
    echo "-f docker-compose.yml -f docker-compose.host-ollama.yml"
  elif [[ "${BASHER_USE_GPU:-0}" == "1" ]]; then
    echo "-f docker-compose.yml -f docker-compose.gpu.yml"
  else
    echo "-f docker-compose.yml"
  fi
}

check_host_ollama() {
  local code
  code=$(curl -s --connect-timeout 2 --max-time 5 -o /dev/null -w "%{http_code}" \
    http://127.0.0.1:11434/api/tags 2>/dev/null) || true
  if [[ "$code" == "200" ]]; then
    return 0
  fi
  echo "Basher: host Ollama not reachable at 127.0.0.1:11434 (HTTP ${code:-000})." >&2
  echo "Basher: start it (e.g. systemctl --user start ollama) or unset BASHER_USE_HOST_OLLAMA." >&2
  return 1
}

check_docker_running() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "Basher: Docker is not running. Trying to start the docker service..." >&2
  if command -v systemctl >/dev/null 2>&1 && sudo systemctl start docker; then
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
      sleep 1
      if docker info >/dev/null 2>&1; then
        return 0
      fi
    done
  fi
  echo "Basher: Docker is not available. Start Docker and retry." >&2
  return 1
}

cmd_start() {
  check_docker_running || exit 2
  cd "$BASHER_INSTALL_DIR" || exit 1
  # UID is readonly in bash; pass it only to the compose child env.
  # shellcheck disable=SC2086
  local files
  files=$(compose_files)

  if [[ "${BASHER_USE_HOST_OLLAMA:-0}" == "1" ]]; then
    if [[ "${BASHER_USE_GPU:-0}" == "1" ]]; then
      echo "Basher: BASHER_USE_HOST_OLLAMA=1 set — ignoring BASHER_USE_GPU (host GPU via host Ollama)."
    fi
    echo "Basher: starting with host Ollama (docker-compose.host-ollama.yml)"
    check_host_ollama || exit 2
    # Stop compose LLM if a previous CPU/GPU start left it on :11434.
    docker compose -f docker-compose.yml stop ollama >/dev/null 2>&1 || true
    # Base api still lists ollama in depends_on (merge); start qdrant then api --no-deps.
    # shellcheck disable=SC2086
    if ! env UID="$(id -u)" GID="$(id -g)" docker compose $files up -d --build qdrant; then
      echo "Basher: failed to start qdrant." >&2
      exit 1
    fi
    # shellcheck disable=SC2086
    if env UID="$(id -u)" GID="$(id -g)" docker compose $files up -d --build --no-deps api; then
      echo "Basher: qdrant + api started (using host Ollama at 127.0.0.1:11434)."
      exit 0
    fi
    echo "Basher: docker compose up (api) failed." >&2
    exit 1
  fi

  if [[ "${BASHER_USE_GPU:-0}" == "1" ]]; then
    echo "Basher: starting with GPU overlay (docker-compose.gpu.yml)"
  else
    echo "Basher: starting without GPU reservation (set BASHER_USE_GPU=1 for NVIDIA,"
    echo "         or BASHER_USE_HOST_OLLAMA=1 for host Ollama)."
  fi
  # bash treats UID as readonly; use env(1) so compose gets UID/GID without
  # assigning the shell variable.
  # shellcheck disable=SC2086
  if env UID="$(id -u)" GID="$(id -g)" docker compose $files up -d --build; then
    echo "Basher: containers started."
    exit 0
  fi
  echo "Basher: docker compose up failed." >&2
  exit 1
}

cmd_stop() {
  check_docker_running || exit 2
  cd "$BASHER_INSTALL_DIR" || exit 1
  # shellcheck disable=SC2086
  local files
  files=$(compose_files)
  # shellcheck disable=SC2086
  if docker compose $files stop; then
    echo "Basher: containers stopped."
    exit 0
  fi
  echo "Basher: docker compose stop failed." >&2
  exit 1
}

cmd_down() {
  check_docker_running || exit 2
  cd "$BASHER_INSTALL_DIR" || exit 1
  # shellcheck disable=SC2086
  local files
  files=$(compose_files)
  # shellcheck disable=SC2086
  if docker compose $files down; then
    echo "Basher: containers removed (volumes kept)."
    exit 0
  fi
  echo "Basher: docker compose down failed." >&2
  exit 1
}

cmd_status() {
  if check_connection; then
    echo "Basher: API is reachable and healthy."
  else
    echo "Basher: API is not reachable or not healthy." >&2
    exit 2
  fi
  exit 0
}

# --- Health check: API must be reachable and health endpoint must report "on" or "ok" ---

check_connection() {
  local health_body health_code
  health_body=$(mktemp)
  health_code=$(curl -s --connect-timeout "$BASHER_CONNECT_TIMEOUT" --max-time "$BASHER_HEALTH_MAX_TIME" \
    -w "%{http_code}" -o "$health_body" -X GET "${API_URL}/health" 2>/dev/null) || true
  if [[ "$health_code" != "200" ]]; then
    rm -f "$health_body"
    if [[ -z "$health_code" || "$health_code" == "000" ]]; then
      echo "Error: Could not reach the API (connection failed or timed out). Is the API running at ${API_URL}?" >&2
    else
      echo "Error: API unreachable (HTTP $health_code). Is the API running at ${API_URL}?" >&2
    fi
    return 1
  fi
  local status
  status=$(jq -r '.status // empty' "$health_body" 2>/dev/null || true)
  rm -f "$health_body"
  if [[ -z "$status" ]]; then
    echo "Error: API health endpoint did not return a valid status. Expected JSON with .status (e.g. \"ok\" or \"on\")." >&2
    return 1
  fi
  # Accept "ok", "on", or similar
  if [[ "$status" != "ok" && "$status" != "on" ]]; then
    echo "Error: API is not ready (health status: $status)." >&2
    return 1
  fi
  return 0
}

# Restore terminal on exit (reset attributes; don't use rmcup — we never use alternate screen, and rmcup can send sequences that move cursor to top on some terminals)
restore_tty() {
  if [[ -t 1 ]]; then
    printf '\n'
    tput sgr0 2>/dev/null
  fi
  rm -f "${tmp:-}"
}
trap restore_tty EXIT INT TERM

# --- Error reporting based on HTTP code from /ask ---

report_http_error() {
  local code="$1"
  local body="$2"
  case "$code" in
    000) echo "Basher: Could not connect to the API. Is it running at ${API_URL}?" >&2 ;;
    401) echo "Basher: API returned 401 Unauthorized." >&2 ;;
    404) echo "Basher: API returned 404 Not Found. Is the /ask endpoint available?" >&2 ;;
    500) echo "Basher: API server error (500). Check the API logs." >&2 ;;
    502|503) echo "Basher: API unavailable ($code). Service may be starting or overloaded." >&2 ;;
    *)   echo "Basher: API returned HTTP $code." >&2 ;;
  esac
  if [[ -n "$body" ]]; then
    local detail
    detail=$(echo "$body" | jq -r '.detail // .message // empty' 2>/dev/null)
    if [[ -n "$detail" ]]; then
      echo "Basher: $detail" >&2
    else
      echo "$body" | head -5 >&2
    fi
  fi
}

# --- Prettify response for terminal ---

prettify() {
  local width="${COLUMNS:-80}"
  if [[ -n "$BASHER_USE_GLOW" ]] && command -v glow >/dev/null 2>&1; then
    echo "$1" | glow - 2>/dev/null || echo "$1" | fold -s -w "$width"
  else
    echo "$1" | fold -s -w "$width" | while IFS= read -r line; do
      if [[ "$line" =~ ^[\ \t]*\`\`\` ]]; then
        printf '\033[36m%s\033[0m\n' "$line"
      elif [[ "$line" =~ ^[\ \t]*#+ ]]; then
        printf '\033[1;34m%s\033[0m\n' "$line"
      else
        printf '%s\n' "$line"
      fi
    done
  fi
}

# --- Agent endpoint (single round-trip) ---
#
# We deliberately do NOT loop /agent/tool-results in the CLI. Tool approval
# is the whole point of agent mode and a one-shot bash wrapper has no good
# UX for it (no key polling, no editing, no streaming feedback). If the
# agent asks for tools, we surface them and exit; the desktop UI is the
# right place for the interactive loop.

run_agent_once() {
  local query="$1"
  local debug="$2"
  local body
  body=$(jq -n --arg q "$query" --argjson d "${debug:-false}" \
    '{query: $q, mode: "interactive", debug: $d, base_model: false}')

  local http_code
  tmp=${tmp:-$(mktemp)}
  http_code=$(curl -s --connect-timeout "$BASHER_CONNECT_TIMEOUT" --max-time "$BASHER_ASK_MAX_TIME" \
    -w "%{http_code}" -o "$tmp" -X POST "${API_URL}/agent/run" \
    -H "Content-Type: application/json" \
    -d "$body" 2>/dev/null) || true
  json=$(cat "$tmp")
  [[ -z "$http_code" ]] && http_code="000"
  if [[ "$http_code" != "200" ]]; then
    report_http_error "$http_code" "$json"
    exit 1
  fi
  if ! echo "$json" | jq -e . >/dev/null 2>&1; then
    echo "Basher: API returned invalid JSON." >&2
    exit 1
  fi

  local final
  final=$(echo "$json" | jq -r '.final_answer // empty')
  if [[ -n "$final" ]]; then
    prettify "$final"
    [[ "${final: -1}" != $'\n' ]] && echo
    return 0
  fi

  # No final answer — agent wants tool calls. Show them and exit 3 (not a
  # generic failure): CLI never executes tools; Electron owns approval.
  local n_tools
  n_tools=$(echo "$json" | jq -r '(.tool_calls // []) | length' 2>/dev/null || echo 0)
  printf '\033[33m%s\033[0m\n' "Basher (agent): proposed ${n_tools} shell command(s). CLI is one-shot and will NOT run them."
  printf '\033[33m%s\033[0m\n' "Basher (agent): open src/agent-ui (Electron) to approve/execute. Exit code 3."
  echo "$json" | jq -r '
    .thoughts as $t
    | (if $t != "" then "\nthinking: " + $t else "" end),
      "",
      "proposed commands (not executed):",
      (.tool_calls // [] | to_entries[] | "  [\(.value.danger // "safe")] $ \(.value.command)")
  ' 2>/dev/null
  return 3
}

# --- Main: parse options, then run ask flow ---

main() {
  local query=""
  local debug="false"
  local use_agent="false"

  # Parse options (planned: --start, --stop, --status)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)   show_help ;;
      --start)     cmd_start ;;
      --stop)      cmd_stop ;;
      --down)      cmd_down ;;
      --status)    cmd_status ;;
      --agent)     use_agent="true" ;;
      --)          shift; break ;;
      -*)          echo "Basher: Unknown option: $1. Use --help for usage." >&2; exit 1 ;;
      *)           break ;;
    esac
    shift
  done

  # Read query from args or stdin (only read stdin when piped/redirected, not when TTY)
  if [[ $# -gt 0 ]]; then
    query="$*"
  else
    if [[ -t 0 ]]; then
      query=""
    else
      query=$(cat 2>/dev/null || true)
    fi
  fi

  if [[ -z "$query" ]]; then
    echo "Usage: basher.sh your question here (no quotes needed)" >&2
    echo "   or: echo \"Your question\" | basher.sh" >&2
    echo "Use --help for full usage and options." >&2
    exit 1
  fi


  [[ -n "$BASHER_DEBUG" ]] && debug="true"

  if ! check_connection; then
    exit 2
  fi

  tmp=$(mktemp)
  cleanup() { kill "${curl_pid:-}" 2>/dev/null; exit 130; }
  trap cleanup INT TERM

  if [[ "$use_agent" == "true" ]]; then
    set +e
    run_agent_once "$query" "$debug"
    local agent_rc=$?
    set -e
    exit "$agent_rc"
  fi

  # Call /ask and capture HTTP code (curl -w) and body (-o); use timeouts so we don't hang if API dies
  local http_code
  http_code=$(curl -s --connect-timeout "$BASHER_CONNECT_TIMEOUT" --max-time "$BASHER_ASK_MAX_TIME" \
    -w "%{http_code}" -o "$tmp" -X POST "${API_URL}/ask" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg q "$query" --argjson d "$debug" '{query: $q, debug: $d}')" 2>/dev/null) || true

  json=$(cat "$tmp")

  # Error based on HTTP code first (empty = timeout/failure)
  [[ -z "$http_code" ]] && http_code="000"
  if [[ "$http_code" != "200" ]]; then
    report_http_error "$http_code" "$json"
    exit 1
  fi

  if ! echo "$json" | jq -e . >/dev/null 2>&1; then
    echo "Basher: API returned invalid JSON." >&2
    echo "$json" | head -5 >&2
    exit 1
  fi

  # Check for API error message (e.g. FastAPI validation)
  if echo "$json" | jq -e '.detail' >/dev/null 2>&1; then
    echo "Basher: $(echo "$json" | jq -r '.detail')" >&2
    exit 1
  fi

  # Extract response text and strip control chars (avoid terminal escape sequences from API)
  response=$(echo "$json" | jq -r '.response // empty' | tr -d '\000-\010\013-\037\177')
  if [[ -z "$response" ]]; then
    echo "Basher: No response from API." >&2
    exit 1
  fi

  prettify "$response"
  [[ "${response: -1}" != $'\n' ]] && echo

  # In debug mode, print context and retrieval info below
  if [[ "$debug" == "true" ]]; then
    echo ""
    printf '\033[1m--- retrieval ---\033[0m\n'
    echo "$json" | jq -r '
      .retrieval as $r |
      "chunks_used=\($r.chunks_used // 0)  used_retrieval=\($r.used_retrieval // false)  best_score=\($r.best_score // "n/a")  skipped_low=\($r.skipped_low_confidence // false)",
      (if ($r.query_hints // []) | length > 0 then "query_hints=\(($r.query_hints | join(",")))" else empty end),
      (if $r.status then "status=\($r.status)" else empty end),
      (($r.collections // [])[] |
        if .collection then
          "  \(.collection): " + (
            if .status then .status
            else "hits=\(.hits // 0) used=\(.used // 0) scores=\((.scores // []) | tostring)"
            end
          )
        else empty end),
      (if ($r.global_top // []) | length > 0 then "top hits:" else empty end),
      (($r.global_top // [])[] |
        "  [\(.score)] \(.collection)  \( (.source // .command // "") )")
    ' 2>/dev/null || echo "(could not parse retrieval)"
    context=$(echo "$json" | jq -r '.context // empty')
    if [[ -n "$context" ]]; then
      echo ""
      printf '\033[1m--- context (preview) ---\033[0m\n'
      echo "$context" | head -25
    fi
  fi
}

main "$@"
