#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
RASA_BIN="$VENV_DIR/bin/rasa"
LOG_DIR="$ROOT_DIR/logs"
RUNTIME_DIR="$ROOT_DIR/.runtime"

QDRANT_CONTAINER="${QDRANT_CONTAINER:-chatbot-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
MINIO_CONTAINER="${MINIO_CONTAINER:-chatbot-minio}"
RABBITMQ_CONTAINER="${RABBITMQ_CONTAINER:-chatbot-rabbitmq}"

SYNC_RECREATE="${SYNC_RECREATE:-false}"

ACTION_PID=""

export PYTHONPATH="$ROOT_DIR"

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

write_runtime_state() {
  mkdir -p "$RUNTIME_DIR"
  echo "$QDRANT_CONTAINER" > "$RUNTIME_DIR/qdrant_container"
  echo "$MINIO_CONTAINER" > "$RUNTIME_DIR/minio_container"
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

# ======================================================
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
    echo "[INFO] Installing Python dependencies..."
    "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"
  fi
}

ensure_env_file() {
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "[INFO] .env not found, creating from .env.example"
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "[WARN] Please review .env values before production usage."
  fi
}

# ======================================================
# Infrastructure containers
# ======================================================
ensure_container() {
  local name="$1"
  local image="$2"
  shift 2
  local ports=("$@")

  if docker ps --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Container $name is already running."
    return
  fi

  if docker ps -a --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Starting existing container: $name"
    docker start "$name" >/dev/null
    sleep 2
    return
  fi

  echo "[INFO] Creating container: $name"
  docker run -d --name "$name" --restart unless-stopped "${ports[@]}" "$image" >/dev/null
  sleep 3
}

ensure_postgres() {
  local name="chatbot-postgres"
  local pg_port="${DB_PORT:-5432}"

  if (echo > /dev/tcp/127.0.0.1/"$pg_port") >/dev/null 2>&1; then
    echo "[INFO] PostgreSQL is already running on port $pg_port."
    return
  fi

  ensure_container "$name" "postgres:15" \
    -e "POSTGRES_USER=${DB_USER:-chatbot_user}" \
    -e "POSTGRES_PASSWORD=${DB_PASSWORD:-supersecret}" \
    -e "POSTGRES_DB=${DB_NAME:-chatbot}" \
    -p "$pg_port":5432 \
    -v chatbot_pg_data:/var/lib/postgresql/data
}

ensure_qdrant() {
  local port="${QDRANT_PORT:-6333}"
  if (echo > /dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    echo "[INFO] Qdrant is already running on port $port."
    return
  fi

  ensure_container "$QDRANT_CONTAINER" "$QDRANT_IMAGE" \
    -p "$port":6333 \
    -p "${QDRANT_GRPC_PORT:-6334}":6334 \
    -v "${QDRANT_VOLUME:-chatbot_qdrant_data}":/qdrant/storage
}

ensure_minio() {
  local port="${MINIO_PORT:-9000}"
  if (echo > /dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    echo "[INFO] MinIO is already running on port $port."
    return
  fi

  ensure_container "$MINIO_CONTAINER" "minio/minio:latest" \
    -p "$port":9000 \
    -p 9001:9001 \
    -e "MINIO_ROOT_USER=${MINIO_ACCESS_KEY:-admin}" \
    -e "MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY:-password123}" \
    -v chatbot_minio_data:/data \
    "server" "/data" "--console-address" ":9001"
}

ensure_rabbitmq() {
  ensure_container "$RABBITMQ_CONTAINER" "rabbitmq:3-management" \
    -p "${RABBITMQ_PORT:-5672}":5672 \
    -p "${RABBITMQ_MGM_PORT:-15672}":15672
}

# ======================================================
# Database schema & data
# ======================================================
ensure_database_ready() {
  echo "[INFO] Checking PostgreSQL table: destinations"

  if "$PYTHON_BIN" - <<'PY' 2>/dev/null; then
from database.db_connection import get_connection
conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT to_regclass('public.destinations')")
exists = cur.fetchone()[0] is not None
cur.close()
conn.close()
raise SystemExit(0 if exists else 1)
PY
    echo "[INFO] PostgreSQL is ready (table destinations exists)"
    return
  fi

  echo "[WARN] Schema not found. Initializing database..."
  "$PYTHON_BIN" -m database.init_db
  "$PYTHON_BIN" -m database.importData
  "$PYTHON_BIN" -m database.importTour
  echo "[INFO] Database initialized."
}

# ======================================================
# Qdrant sync
# ======================================================
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

# ======================================================
generate_endpoints() {
  echo "[INFO] Generating endpoints.yml from environment variables..."
  "$PYTHON_BIN" -m scripts.generate_endpoints
}

start_actions() {
  generate_endpoints
  mkdir -p "$LOG_DIR"
  mkdir -p "$RUNTIME_DIR"
  echo "[INFO] Starting action server (logs/actions.log)"
  (
    cd "$ROOT_DIR/rasa_bot"
    "$RASA_BIN" run actions > "$LOG_DIR/actions.log" 2>&1
  ) &
  ACTION_PID=$!
  write_runtime_state
  sleep 2
}

start_shell() {
  generate_endpoints
  echo "[INFO] Starting Rasa shell..."
  cd "$ROOT_DIR/rasa_bot"
  "$RASA_BIN" shell
}

# ======================================================
# MAIN
# ======================================================
main() {
  write_runtime_state
  ensure_tools
  ensure_venv
  ensure_env_file
  ensure_postgres
  ensure_qdrant
  ensure_minio
  ensure_rabbitmq
  ensure_database_ready
  sync_qdrant
  start_actions
  start_shell
}

main "$@"
