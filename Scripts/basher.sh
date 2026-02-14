#!/bin/bash
# Basher CLI: ask the Fedora bash assistant.
# Usage: basher.sh hello whats up   or   basher.sh "one quoted question"   or   echo "question" | basher.sh

set -e
API_URL="${BASHER_API_URL:-http://localhost:8000}"

# Restore terminal on any exit (prevents crash/bad state when script exits or is killed)
restore_tty() {
  if [[ -t 1 ]]; then
    printf '\r\n'
    tput sgr0 2>/dev/null
    tput rmcup 2>/dev/null
  fi
  rm -f "${tmp:-}"
}
trap restore_tty EXIT INT TERM

# Read query from args (natural: basher hello whats up) or stdin
if [[ -n "$*" ]]; then
  query="$*"
else
  query=$(cat)
fi

if [[ -z "$query" ]]; then
  echo "Usage: basher.sh your question here (no quotes needed)"
  echo "   or: echo \"Your question\" | basher.sh"
  exit 1
fi

# Optional: debug mode (set BASHER_DEBUG=1 to see context and retrieval)
debug="false"
[[ -n "$BASHER_DEBUG" ]] && debug="true"

# Call API in background
tmp=$(mktemp)
cleanup() { kill "${curl_pid:-}" 2>/dev/null; exit 130; }
trap cleanup INT TERM
curl -s -X POST "${API_URL}/ask" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg q "$query" --argjson d "$debug" '{query: $q, debug: $d}')" \
  -o "$tmp" &
curl_pid=$!

# Spinner in foreground: \ | / - until curl finishes (so it always shows)
spin=($'\\' '|' '/' '-')
i=0
while kill -0 "$curl_pid" 2>/dev/null; do
  printf '\r  %s  ' "${spin[i]}"
  i=$(( (i + 1) % 4 ))
  sleep 0.12
done
wait "$curl_pid" 2>/dev/null || true
printf '\r    \r\n'
json=$(cat "$tmp")

if ! echo "$json" | jq -e . >/dev/null 2>&1; then
  echo "Basher: API error or invalid JSON." >&2
  echo "$json" | head -5 >&2
  exit 1
fi

# Check for API error message
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

# Prettify for terminal. Use glow only if BASHER_USE_GLOW=1 (glow can leave terminal in bad state).
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

prettify "$response"
# Ensure terminal ends on a newline so the next prompt is clean
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
