#!/bin/sh
set -e
# Start ollama in background and pull model on first run (visible progress, no build-time download)
ollama serve &
OLLAMA_PID=$!
# Give the server a moment to bind before we poll (no curl needed)
sleep 3
# Wait for API using ollama CLI (max 90s)
TIMEOUT=90
ELAPSED=0
until ollama list > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo "Ollama did not become ready in ${TIMEOUT}s"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done
# Pull model if requested (shows progress at container start)
if [ -n "$OLLAMA_MODEL" ]; then
  ollama pull "$OLLAMA_MODEL" || true
fi
# Keep ollama in foreground so the container stays up
wait $OLLAMA_PID
