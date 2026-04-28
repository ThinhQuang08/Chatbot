#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
RASA_BIN="$VENV_DIR/bin/rasa"
LOG_DIR="$ROOT_DIR/logs"
RUNTIME_DIR="$ROOT_DIR/.runtime"

QDRANT_CONTAINER="${QDRANT_CONTAINER:-chatbot-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
QDRANT_VOLUME="${QDRANT_VOLUME:-chatbot_qdrant_data}"

MODEL_FILE="${MODEL_FILE:-level45-level5-v2.tar.gz}"
SYNC_RECREATE="${SYNC_RECREATE:-false}" # true = tạo mới dữ liệu

ACTION_PID=""

write_runtime_state() {
  mkdir -p "$RUNTIME_DIR"
  echo "$QDRANT_CONTAINER" > "$RUNTIME_DIR/qdrant_container"
  if [[ -n "$ACTION_PID" ]]; then
    echo "$ACTION_PID" > "$RUNTIME_DIR/action_server.pid"
  fi
}

remove_runtime_state() {
  rm -f "$RUNTIME_DIR/action_server.pid"
}

cleanup() {
  if [[ -n "$ACTION_PID" ]] && kill -0 "$ACTION_PID" 2>/dev/null; then
    kill "$ACTION_PID" 2>/dev/null || true
  fi
  remove_runtime_state
}
trap cleanup EXIT INT TERM

ensure_tools() {
  command -v docker >/dev/null 2>&1 || {
    echo "[ERROR] Docker is required but not found."
    exit 1
  }

  command -v python3 >/dev/null 2>&1 || {
    echo "[ERROR] python3 is required but not found."
    exit 1
  }
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  echo "[INFO] Installing Python dependencies"
  # "$PIP_BIN" install --upgrade pip
  # "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

ensure_env_file() {
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "[INFO] .env not found, creating from .env.example"
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "[WARN] Please review .env values before production usage."
  fi
}

ensure_database_ready() {
  echo "[INFO] Checking PostgreSQL table: destinations"

  if "$PYTHON_BIN" - <<'PY'
from database.db_connection import get_connection

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT to_regclass('public.destinations')")
exists = cur.fetchone()[0] is not None
cur.close()
conn.close()
raise SystemExit(0 if exists else 1)
PY
  then
    echo "[INFO] PostgreSQL is ready (table destinations exists)"
    return
  fi

  echo "[WARN] Schema not found or incomplete. Initializing Database Schema..."
  "$PYTHON_BIN" -m database.init_db

  echo "[WARN] Table destinations not found. Importing initial dataset..."
  #"$PYTHON_BIN" -m database.import_data
  "$PYTHON_BIN" -m database.importData

  echo "[INFO] Importing Tour dataset into PostgreSQL..."
  #"$PYTHON_BIN" -m database.import_tour
  "$PYTHON_BIN" -m database.importTour

  echo "[INFO] Imported initial dataset into PostgreSQL"
}

ensure_qdrant() {
  if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
    echo "[INFO] Qdrant container is already running: $QDRANT_CONTAINER"
    return
  fi

  if docker ps -a --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
    echo "[INFO] Starting existing Qdrant container: $QDRANT_CONTAINER"
    docker start "$QDRANT_CONTAINER" >/dev/null
    return
  fi

  echo "[INFO] Creating and starting Qdrant container: $QDRANT_CONTAINER"
  docker run -d \
    --name "$QDRANT_CONTAINER" \
    -p "$QDRANT_PORT":6333 \
    -p "$QDRANT_GRPC_PORT":6334 \
    -v "$QDRANT_VOLUME":/qdrant/storage \
    "$QDRANT_IMAGE" >/dev/null

  write_runtime_state
}

sync_qdrant() {
  echo "[INFO] Qdrant healthcheck"
  "$PYTHON_BIN" -m scripts.check_qdrant || true

  echo "[INFO] Syncing vectors into Qdrant"
  if [[ "$SYNC_RECREATE" == "true" ]]; then
    "$PYTHON_BIN" -m scripts.sync_qdrant --recreate
    "$PYTHON_BIN" -m scripts.sync_qdrant_tour --recreate
  else
    "$PYTHON_BIN" -m scripts.sync_qdrant
    "$PYTHON_BIN" -m scripts.sync_qdrant_tour
  fi
}

start_actions() {
  mkdir -p "$LOG_DIR"
  mkdir -p "$RUNTIME_DIR"
  echo "[INFO] Starting action server (logs/actions.log)"
  (
    cd "$ROOT_DIR/rasa_bot"
    "$RASA_BIN" run actions >"$LOG_DIR/actions.log" 2>&1
  ) &
  ACTION_PID=$!
  write_runtime_state
  sleep 2
}

start_shell() {
  local model_path="$ROOT_DIR/rasa_bot/models/$MODEL_FILE"
  if [[ ! -f "$model_path" ]]; then
    echo "[ERROR] Model not found: $model_path"
    echo "[INFO] Train a model first, e.g."
    # echo "       $RASA_BIN train --fixed-model-name level45-level5-v2"
    echo "       $RASA_BIN train"
    exit 1
  fi

  echo "[INFO] Starting Rasa shell with model: $MODEL_FILE"
  cd "$ROOT_DIR/rasa_bot"
  # "$RASA_BIN" shell --model "models/$MODEL_FILE"
  "$RASA_BIN" shell
}

main() {
  write_runtime_state
  ensure_tools
  ensure_venv
  ensure_env_file
  ensure_database_ready
  ensure_qdrant
  sync_qdrant
  start_actions
  start_shell
}

main "$@"
