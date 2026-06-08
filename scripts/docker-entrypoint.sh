#!/bin/bash
set -e

ROOT_DIR="/app"
cd "$ROOT_DIR"

echo "[ENTRYPOINT] Generating endpoints.yml from environment variables..."
python -m scripts.generate_endpoints

cd "$ROOT_DIR/rasa_bot"

cleanup() {
  echo "[ENTRYPOINT] Shutting down..."
  kill $ACTION_PID $RASA_PID 2>/dev/null || true
  wait $ACTION_PID $RASA_PID 2>/dev/null || true
  echo "[ENTRYPOINT] Goodbye."
}
trap cleanup EXIT INT TERM

echo "[ENTRYPOINT] Starting action server on port 5055..."
rasa run actions --port 5055 &
ACTION_PID=$!

echo "[ENTRYPOINT] Waiting for action server..."
until curl -s http://localhost:5055/health | grep -q '"status":"ok"'; do
  sleep 1
done
echo "[ENTRYPOINT] Action server is ready."

echo "[ENTRYPOINT] Starting Rasa API server on port 5005..."
rasa run \
  --enable-api \
  --cors "*" \
  --endpoints endpoints.yml \
  --port 5005 &
RASA_PID=$!

echo "[ENTRYPOINT] Waiting for Rasa API server..."
for i in $(seq 1 30); do
  if curl -s http://localhost:5005/ > /dev/null 2>&1; then
    echo "[ENTRYPOINT] Rasa API server is ready."
    break
  fi
  sleep 1
done

echo "[ENTRYPOINT] All servers running."
wait -n
