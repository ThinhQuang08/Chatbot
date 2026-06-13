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
MINIO_CONTAINER="${MINIO_CONTAINER:-chatbot-minio}"
MINIO_IMAGE="${MINIO_IMAGE:-minio/minio:latest}"
RABBITMQ_CONTAINER="${RABBITMQ_CONTAINER:-chatbot-rabbitmq}"

MLFLOW_PID=""

export PYTHONPATH="$ROOT_DIR"

# Load .env vào environment
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

write_runtime_state() {
  mkdir -p "$RUNTIME_DIR"
  echo "$QDRANT_CONTAINER" > "$RUNTIME_DIR/qdrant_container"
  echo "$MINIO_CONTAINER" > "$RUNTIME_DIR/minio_container"
  if [[ -n "$MLFLOW_PID" ]]; then
    echo "$MLFLOW_PID" > "$RUNTIME_DIR/mlflow.pid"
  fi
}

cleanup() {
  echo ""
  echo "[INFO] Dọn dẹp các tiến trình phụ..."
  if [[ -n "$MLFLOW_PID" ]] && kill -0 "$MLFLOW_PID" 2>/dev/null; then
    kill "$MLFLOW_PID" 2>/dev/null || true
  fi
  rm -f "$RUNTIME_DIR/mlflow.pid"
}
trap cleanup EXIT INT TERM

echo "==================================================="
echo "   CHUẨN BỊ MÔI TRƯỜNG DÀNH CHO USER MỚI QUA GIT   "
echo "==================================================="

# =========================================================
# 1. Kiểm tra Docker
# =========================================================
ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker không tồn tại. Vui lòng cài đặt Docker."
    exit 1
  fi
  echo "[INFO] Docker OK"
}

# =========================================================
# 2. PostgreSQL
# =========================================================
ensure_postgres() {
  local name="chatbot-postgres"
  local pg_port="${DB_PORT:-5432}"
  if (echo > /dev/tcp/127.0.0.1/"$pg_port") >/dev/null 2>&1; then
    echo "[INFO] PostgreSQL đã chạy sẵn ở cổng $pg_port. Bỏ qua Docker."
    return
  fi
  if docker ps --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Container $name đã đang chạy."
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Khởi động lại container $name..."
    docker start "$name" >/dev/null
    sleep 3
    return
  fi
  echo "[INFO] Khởi tạo container PostgreSQL..."
  docker run -d \
    --name "$name" \
    -e POSTGRES_USER="${DB_USER:-chatbot_user}" \
    -e POSTGRES_PASSWORD="${DB_PASSWORD:-supersecret}" \
    -e POSTGRES_DB="${DB_NAME:-chatbot}" \
    -p "$pg_port":5432 \
    -v chatbot_pg_data:/var/lib/postgresql/data \
    postgres:15 >/dev/null
  echo "[INFO] Đợi 5 giây để PostgreSQL khởi tạo..."
  sleep 5
}

# =========================================================
# 3. Qdrant
# =========================================================
ensure_qdrant() {
  local name="$QDRANT_CONTAINER"
  local port="${QDRANT_PORT:-6333}"
  if (echo > /dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    echo "[INFO] Qdrant đã chạy sẵn ở cổng $port. Bỏ qua Docker."
    return
  fi
  if docker ps --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Qdrant container đã đang chạy."
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Khởi động lại Qdrant container..."
    docker start "$name" >/dev/null
    sleep 2
    return
  fi
  echo "[INFO] Khởi tạo container Qdrant..."
  docker run -d \
    --name "$name" \
    --restart unless-stopped \
    -p "$port":6333 \
    -p "${QDRANT_GRPC_PORT:-6334}":6334 \
    -v "${QDRANT_VOLUME:-chatbot_qdrant_data}":/qdrant/storage \
    "$QDRANT_IMAGE" >/dev/null
  echo "[INFO] Đợi 3 giây để Qdrant sẵn sàng..."
  sleep 3
}

# =========================================================
# 4. MinIO
# =========================================================
ensure_minio() {
  local name="$MINIO_CONTAINER"
  local port="${MINIO_PORT:-9000}"
  if (echo > /dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    echo "[INFO] MinIO đã chạy sẵn ở cổng $port. Bỏ qua Docker."
    # Vẫn set bucket policy đề phòng
    return
  fi
  if docker ps --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] MinIO container đã đang chạy."
    setup_minio_bucket
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Khởi động lại MinIO container..."
    docker start "$name" >/dev/null
    sleep 3
    setup_minio_bucket
    return
  fi
  echo "[INFO] Khởi tạo container MinIO..."
  docker run -d \
    --name "$name" \
    --restart unless-stopped \
    -p "$port":9000 \
    -p 9001:9001 \
    -e MINIO_ROOT_USER="${MINIO_ACCESS_KEY:-admin}" \
    -e MINIO_ROOT_PASSWORD="${MINIO_SECRET_KEY:-password123}" \
    -v chatbot_minio_data:/data \
    "$MINIO_IMAGE" server /data --console-address ":9001" >/dev/null
  echo "[INFO] Đợi 5 giây để MinIO sẵn sàng..."
  sleep 5
  setup_minio_bucket
  write_runtime_state
}

setup_minio_bucket() {
  local mc_image="minio/mc:latest"
  local minio_url="${MINIO_URL:-http://localhost:9000}"
  local access_key="${MINIO_ACCESS_KEY:-admin}"
  local secret_key="${MINIO_SECRET_KEY:-password123}"
  local bucket="${MINIO_BUCKET:-chatbot-models}"

  # Xoá host protocol để dùng trong mc alias
  local minio_host="${minio_url#http://}"
  minio_host="${minio_host#https://}"

  echo "[INFO] Tạo bucket MinIO: $bucket"
  docker run --rm --network host "$mc_image" alias set myminio http://"$minio_host" "$access_key" "$secret_key" >/dev/null 2>&1 || true

  # Tạo bucket (ignore error nếu đã tồn tại)
  docker run --rm --network host "$mc_image" mb "myminio/$bucket" >/dev/null 2>&1 || true

  echo "[INFO] Set policy public read cho bucket $bucket"
  docker run --rm --network host "$mc_image" anonymous set public "myminio/$bucket" >/dev/null 2>&1 || true

  echo "[INFO] MinIO bucket ready: $minio_url/$bucket"
}

# =========================================================
# 5. MLflow (local process)
# =========================================================
ensure_mlflow() {
  if (echo > /dev/tcp/127.0.0.1/"${MLFLOW_PORT:-5000}") >/dev/null 2>&1; then
    echo "[INFO] MLflow đã chạy sẵn ở cổng ${MLFLOW_PORT:-5000}."
    return
  fi
  echo "[INFO] Khởi động MLflow server local..."
  mkdir -p "$ROOT_DIR/mlruns"
  nohup "$PYTHON_BIN" -m mlflow server \
    --host 0.0.0.0 \
    --port "${MLFLOW_PORT:-5000}" \
    --backend-store-uri "sqlite:///$ROOT_DIR/mlflow.db" \
    --default-artifact-root "$ROOT_DIR/mlruns" \
    > "$LOG_DIR/mlflow.log" 2>&1 &
  MLFLOW_PID=$!
  write_runtime_state
  echo "[INFO] MLflow PID $MLFLOW_PID (log: logs/mlflow.log)"
  sleep 2
}

# =========================================================
# 6. RabbitMQ
# =========================================================
ensure_rabbitmq() {
  local name="$RABBITMQ_CONTAINER"
  if docker ps --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] RabbitMQ container đã đang chạy."
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -q "^$name$"; then
    echo "[INFO] Khởi động lại RabbitMQ container..."
    docker start "$name" >/dev/null
    sleep 5
    return
  fi
  echo "[INFO] Khởi tạo container RabbitMQ..."
  docker run -d \
    --name "$name" \
    --restart unless-stopped \
    -p "${RABBITMQ_PORT:-5672}":5672 \
    -p "${RABBITMQ_MGM_PORT:-15672}":15672 \
    rabbitmq:3-management >/dev/null
  echo "[INFO] Đợi 5 giây để RabbitMQ sẵn sàng..."
  sleep 5
}

# =========================================================
# 7. Virtual environment
# =========================================================
ensure_venv() {
  echo "[INFO] Cài đặt Python Virtual Environment..."
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  "$PIP_BIN" install --upgrade pip >/dev/null
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

# =========================================================
# 8. File .env
# =========================================================
ensure_env_file() {
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "[INFO] Tạo file .env từ .env.example..."
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "[WARN] Đã tạo .env mặc định. Kiểm tra và chỉnh sửa nếu cần."
  fi
  # Load lại .env sau khi có file
  set -a
  source "$ROOT_DIR/.env"
  set +a
}

# =========================================================
# 9. Khởi tạo Database schema & Import dữ liệu
# =========================================================
ensure_database_ready() {
  echo "[INFO] Kiểm tra database..."
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
    echo "[INFO] Database đã sẵn sàng (table destinations tồn tại)."
    return
  fi

  echo "[INFO] Khởi tạo database schema..."
  "$PYTHON_BIN" -m database.init_db
  echo "[INFO] Import dữ liệu destinations..."
  "$PYTHON_BIN" -m database.importData
  echo "[INFO] Import dữ liệu tours..."
  "$PYTHON_BIN" -m database.importTour
  echo "[INFO] Import dữ liệu accommodations..."
  "$PYTHON_BIN" -m database.import_accommodations || true
}

# =========================================================
# 10. Sync Qdrant vectors
# =========================================================
ensure_qdrant_sync() {
  echo "[INFO] Sync embeddings lên Qdrant..."
  "$PYTHON_BIN" -m scripts.check_qdrant || true
  "$PYTHON_BIN" -m scripts.sync_qdrant --recreate
  "$PYTHON_BIN" -m scripts.sync_qdrant_tour --recreate
}

# =========================================================
# 11. Train model & push lên MinIO
# =========================================================
train_model() {
  echo "[INFO] Train model (train_mlflow.py)..."
  "$PYTHON_BIN" -m scripts.train_mlflow
}

# =========================================================
# 12. Chạy Action Server + Rasa shell
# =========================================================
start_stack() {
  echo "[INFO] Generating endpoints.yml..."
  "$PYTHON_BIN" -m scripts.generate_endpoints

  echo "[INFO] Starting action server (logs/actions.log)..."
  mkdir -p "$LOG_DIR"
  mkdir -p "$RUNTIME_DIR"
  (
    cd "$ROOT_DIR/rasa_bot"
    "$RASA_BIN" run actions > "$LOG_DIR/actions.log" 2>&1
  ) &
  ACTION_PID=$!
  echo "$ACTION_PID" > "$RUNTIME_DIR/action_server.pid"
  sleep 3

  echo "[INFO] Mở Rasa shell để test..."
  cd "$ROOT_DIR/rasa_bot"
  "$RASA_BIN" shell
}

# =========================================================
# MAIN
# =========================================================
main() {
  ensure_docker
  ensure_postgres
  ensure_qdrant
  ensure_minio
  ensure_venv
  ensure_env_file
  ensure_mlflow
  ensure_rabbitmq
  ensure_database_ready
  ensure_qdrant_sync
  train_model
  start_stack
}

main "$@"
