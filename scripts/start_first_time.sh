#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
MODEL_FILE="${MODEL_FILE:-level45-level5-v2.tar.gz}"

echo "==================================================="
echo "   CHUẨN BỊ MÔI TRƯỜNG DÀNH CHO USER MỚI QUA GIT   "
echo "==================================================="

# 1. Kiểm tra Docker
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker không tồn tại. Vui lòng cài đặt Docker để chạy PostgreSQL và Qdrant."
    exit 1
fi

# 2. Khởi tạo/Bật PostgreSQL
if (echo > /dev/tcp/127.0.0.1/5432) >/dev/null 2>&1; then
    echo "[INFO] Phát hiện dịch vụ PostgreSQL (hoặc tiến trình khác) đang chạy sẵn ở cổng 5432. Bỏ qua khởi tạo bằng Docker."
elif docker ps --format '{{.Names}}' | grep -q "^chatbot-postgres$"; then
    echo "[INFO] Container chatbot-postgres đã đang chạy."
elif docker ps -a --format '{{.Names}}' | grep -q "^chatbot-postgres$"; then
    echo "[INFO] Khởi động lại container chatbot-postgres..."
    docker start chatbot-postgres >/dev/null
    sleep 3
else
    echo "[INFO] Khởi tạo container PostgreSQL lần đầu..."
    docker run -d \
        --name chatbot-postgres \
        -e POSTGRES_USER=chatbot_user \
        -e POSTGRES_PASSWORD=supersecret \
        -e POSTGRES_DB=chatbot \
        -p 5432:5432 \
        -v chatbot_pg_data:/var/lib/postgresql/data \
        postgres:15 >/dev/null
    echo "[INFO] Đợi 5 giây để PostgreSQL cấu hình xong..."
    sleep 5
fi

# 3. Tạo virtual environment và cài dependencies
echo "[INFO] Cài đặt Python Virtual Environment và Dependencies (quá trình này có thể tốn vài phút tùy mạng)..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi

"$PIP_BIN" install --upgrade pip >/dev/null
"$PIP_BIN" install -r "$ROOT_DIR/requirements.txt" >/dev/null

# 4. Sao chép file cấu hình (.env)
if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "[INFO] Tạo file .env từ .env.example..."
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

# 5. Huấn luyện (Train) mô hình Rasa nếu thiếu
MODEL_PATH="$ROOT_DIR/rasa_bot/models/$MODEL_FILE"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[WARN] Không tìm thấy model Rasa: $MODEL_PATH"
    echo "[INFO] Tiến hành huấn luyện (train) mô hình mặc định..."
    (
        cd "$ROOT_DIR/rasa_bot"
        "$VENV_DIR/bin/rasa" train --fixed-model-name "level45-level5-v2"
    )
fi

# 6. Chạy toàn bộ stack
echo "=================================================="
echo "   Môi trường chuẩn bị xong - Khởi chạy Chatbot   "
echo "=================================================="
export SYNC_RECREATE=true
bash "$ROOT_DIR/scripts/start_all.sh"
