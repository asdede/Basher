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
# OPTIONS (planned; currently only --help is implemented)
#   --help    Show this help and exit.
#   --start   Starts the basher containers.
#   --stop    Stops the basher containers.
#   --down    Destroys the basher containers.
#   --status  Show whether the API is running and healthy.
#
# ENVIRONMENT
#   BASHER_API_URL            Base URL of the API (default: http://localhost:8000).
#   BASHER_DEBUG              If set, show context and retrieval info (e.g. BASHER_DEBUG=1).
#   BASHER_USE_GLOW           If set and glow is installed, render markdown with glow.
#   BASHER_CONNECT_TIMEOUT    Seconds to wait for connection (default: 5).
#   BASHER_HEALTH_MAX_TIME    Max seconds for health check (default: 10).
#   BASHER_ASK_MAX_TIME       Max seconds for /ask request (default: 300).
#
# EXAMPLES
#   basher.sh how do I list files
#   basher.sh "one quoted question"
#   echo "how do I use grep?" | basher.sh
#   BASHER_DEBUG=1 basher.sh explain bash arrays
#
# EXIT CODES
#   0   Success.
#   1   Usage error, API error, no response, or invalid JSON.
#   2   API unreachable or not healthy (connection failed or health check failed).
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

# --- Subcommands (stubs for future --start, --stop, etc.) ---

show_help() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
  echo "Usage: basher.sh [OPTIONS] your question here"
  echo "   or: echo \"Your question\" | basher.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --help   Show this help."
  echo "  --start  Starts the basher containers."
  echo "  --stop   Stops the basher containers."
  echo "  --down   Destroys the basher containers."
  echo "  --status Show API status."
  exit 0
}

cmd_start() {
  check_docker_running
  # Always update the containers (Should not fill the cache but might, lets see)
  cd $BASHER_INSTALL_DIR
  docker compose up -d --build

  exit 1
}

cmd_stop() {
  check_docker_running
  cd $BASHER_INSTALL_DIR
  docker compose stop
  exit 1
}

cmd_down() {
  check_docker_running
  cd $BASHER_INSTALL_DIR
  docker compose down
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

check_docker_running() {
  if ! docker ps > /dev/null 2>&1; then
    echo "Docker is not running. Starting docker service..." >&2
    sudo systemctl start docker
    return 1
  fi
}

# --- Main: parse options, then run ask flow ---

main() {
  local query=""
  local debug="false"

  # Parse options (planned: --start, --stop, --status)
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)   show_help ;;
      --start)     cmd_start ;;
      --stop)      cmd_stop ;;
      --status)    cmd_status ;;
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
    context=$(echo "$json" | jq -r '.context // empty')
    if [[ -n "$context" ]]; then
      echo ""
      printf '\033[2m--- context (preview) ---\033[0m\n'
      echo "$context" | head -25
    fi
    echo "$json" | jq -r '.retrieval.collections[]? | "  \(.collection): \(.status // "used=\(.used) scores=\(.scores)")"' 2>/dev/null | sed 's/^/\033[2m/; s/$/\033[0m/' || true
  fi
}

main "$@"
