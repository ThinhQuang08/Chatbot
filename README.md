# Chatbot Du Lịch (Rasa + PostgreSQL + Qdrant + MinIO)

Dự án chatbot tư vấn du lịch tiếng Việt, hỗ trợ cả chạy local và Docker.

## Kiến trúc tổng quan

| Thành phần | Vai trò |
| --- | --- |
| **Rasa** | NLU + hội thoại, serve REST API (port 5005) |
| **Action Server** | Xử lý custom actions (port 5055) |
| **PostgreSQL** | Lưu dữ liệu điểm đến + tracker store |
| **Qdrant** | Vector search cho semantic rerank |
| **MinIO** | Model registry (lưu Rasa model, load tại runtime) |
| **MLflow** | Tracking training metrics & artifacts |
| **RabbitMQ** | Event broker cho conversation events |
| **Dashboard** | Flask app gán nhãn dữ liệu + retrain |

## Yêu cầu

- Linux/macOS (Ubuntu 22.04+ khuyến nghị)
- Python 3.10
- Docker

---

## 1) Cài đặt thủ công từ A-Z

Phần này hướng dẫn từng bước để tự tay chạy tất cả thành phần mà không cần scripts tự động.

### 1.1 Clone & môi trường Python

```bash
git clone https://github.com/ThinhQuang08/Chatbot.git Chatbot
cd Chatbot

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Tạo .env từ mẫu
cp .env.example .env
# Sau đó sửa .env cho phù hợp với môi trường của bạn
```

### 1.2 Khởi động PostgreSQL

```bash
# Tạo Docker volume riêng để dữ liệu không bị mất
docker volume create chatbot_pg_data

# Chạy container
docker run -d \
  --name chatbot-postgres \
  -e POSTGRES_USER=chatbot_user \
  -e POSTGRES_PASSWORD=supersecret \
  -e POSTGRES_DB=chatbot \
  -p 5432:5432 \
  -v chatbot_pg_data:/var/lib/postgresql/data \
  postgres:15

# Kiểm tra
docker ps --filter name=chatbot-postgres
docker exec -it chatbot-postgres psql -U chatbot_user -d chatbot -c "SELECT 1;"
```

Khi cần dừng / xoá:

```bash
docker stop chatbot-postgres
docker rm chatbot-postgres
# Xoá luôn dữ liệu (cẩn thận):
# docker volume rm chatbot_pg_data
```

### 1.3 Khởi tạo Database schema & Import dữ liệu

```bash
# Tạo các bảng (destinations, tours, accommodations, tour_destinations)
python -m database.init_db

# Import dữ liệu điểm đến
python -m database.importData

# Import dữ liệu tour
python -m database.importTour

# Import dữ liệu accommodations (khách sạn)
python -m database.import_accommodations
```

### 1.4 Khởi động Qdrant (Vector Search)

```bash
# Tạo volume cho Qdrant
docker volume create chatbot_qdrant_data

# Chạy container
docker run -d \
  --name chatbot-qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v chatbot_qdrant_data:/qdrant/storage \
  qdrant/qdrant

# Kiểm tra health
curl http://localhost:6333/health
```

### 1.5 Sync embeddings lên Qdrant

```bash
# Sync destinations (tạo collection + upsert vectors)
python -m scripts.sync_qdrant

# Sync tours
python -m scripts.sync_qdrant_tour

# Nếu cần xoá collection cũ và tạo lại từ đầu:
python -m scripts.sync_qdrant --recreate
python -m scripts.sync_qdrant_tour --recreate

# Kiểm tra collection đã có dữ liệu chưa
python -m scripts.check_qdrant
```

### 1.6 Khởi động MinIO (Model Registry)

```bash
# Tạo volume
docker volume create chatbot_minio_data

# Chạy MinIO
docker run -d \
  --name chatbot-minio \
  --restart unless-stopped \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=admin \
  -e MINIO_ROOT_PASSWORD=password123 \
  -v chatbot_minio_data:/data \
  minio/minio server /data --console-address ":9001"

# Kiểm tra
curl http://localhost:9000/minio/health/live
```

**Tạo bucket và set policy public read:**

```bash
# Cài MinIO Client
docker run --rm -it --entrypoint /bin/sh minio/mc -c "apk add --no-cache curl && echo done" > /dev/null

# Dùng mc alias để kết nối
docker run --rm --network host \
  minio/mc alias set myminio http://localhost:9000 admin password123

# Tạo bucket
docker run --rm --network host \
  minio/mc mb myminio/chatbot-models

# Set policy public read (để Rasa có thể GET model qua HTTP)
docker run --rm --network host \
  minio/mc anonymous set public myminio/chatbot-models

# Kiểm tra
curl http://localhost:9000/chatbot-models/
```

### 1.7 Khởi động MLflow Tracking Server

Mở 1 terminal riêng và chạy:

```bash
# Tạo thư mục lưu artifacts
mkdir -p mlruns

# Start MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Giữ terminal này chạy, truy cập `http://localhost:5000` để xem UI.

### 1.8 Khởi động RabbitMQ (Event Broker)

```bash
docker run -d \
  --name chatbot-rabbitmq \
  --restart unless-stopped \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# Kiểm tra (truy cập http://localhost:15672, user: guest, pass: guest)
```

### 1.9 Train Rasa model & push lên MinIO

Sau khi đã có PostgreSQL, Qdrant, MLflow và các dependencies:

```bash
python -m scripts.train_mlflow
```

Script `train_mlflow.py` tự động:

1. Train Rasa model (chạy `rasa train --data data/train`)
2. Cross-validation 3 folds
3. Log metrics (F1, accuracy, precision, recall) lên MLflow
4. Upload artifact lên MLflow
5. Gọi `deploy_model.run_cd_pipeline()` - so sánh F1 với best hiện tại → nếu đạt chuẩn thì push `.tar.gz` lên MinIO bucket `chatbot-models/latest_model.tar.gz`

**Nếu chỉ muốn deploy model hiện có lên MinIO (không train lại):**

```bash
python -m scripts.upload_model_s3
```

### 1.10 Chạy Action Server

```bash
cd rasa_bot

# Gen endpoints.yml từ .env
python -m scripts.generate_endpoints

# Start action server
rasa run actions --port 5055
# (chạy ở background: thêm & hoặc dùng terminal riêng)
```

### 1.11 Chạy Rasa API Server

Trong terminal riêng (sau khi action server đã sẵn sàng):

```bash
cd rasa_bot
rasa run \
  --enable-api \
  --cors "*" \
  --endpoints endpoints.yml \
  --port 5005
```

### 1.12 Kiểm tra

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender":"test","message":"chào bạn, tôi muốn đi du lịch đà lạt"}'
```

### 1.13 Chạy Dashboard (tuỳ chọn)

```bash
python -m admin_dashboard.app
# Truy cập http://localhost:5001
```

### 1.14 Chạy MLflow UI (tuỳ chọn)

Đã chạy sẵn ở bước 1.7, truy cập `http://localhost:5000`.

---

## 2) Chạy bằng script (nhanh hơn)

Sau khi đã nắm rõ các bước thủ công, có thể dùng script để tự động hoá.

### 2.1 Lần đầu tiên (vừa git clone)

```bash
bash scripts/start_first_time.sh
```

Script tự động: tạo venv → cài pip → khởi tạo DB → train model → push MinIO → chạy stack.

### 2.2 Chạy hàng ngày

```bash
bash scripts/start_all.sh
```

Tự động: kiểm tra PostgreSQL → khởi động Qdrant + RabbitMQ → sync vector → gen endpoints.yml → start action server → mở Rasa shell.

### 2.3 Dừng stack

```bash
bash scripts/stop_all.sh
# xoá luôn container Qdrant nếu muốn:
PURGE_QDRANT=true bash scripts/stop_all.sh
```

---

## 3) Chạy Docker (single image Rasa)

Build 1 image duy nhất chứa cả Rasa API + Action Server, model load từ MinIO.

### 3.1 Build image

```bash
docker build -t chatbot-rasa .
```

### 3.2 Chạy container

```bash
docker run -d \
  --name chatbot \
  -p 5005:5005 \
  -p 5055:5055 \
  --env-file .env \
  --add-host host.docker.internal:host-gateway \
  chatbot-rasa
```

Container sẽ: gen endpoints.yml → start action server (bg) → wait health → start Rasa API.

**Lưu ý:** Container cần kết nối tới PostgreSQL, Qdrant, MinIO đang chạy trên host. Dùng `--add-host host.docker.internal:host-gateway` để container gọi được các service qua `host.docker.internal`. File `.env.docker` có sẵn các host trỏ về `host.docker.internal`.

### 3.3 Push lên Docker Hub

```bash
docker login
docker tag chatbot-rasa <username>/chatbot-rasa:latest
docker push <username>/chatbot-rasa:latest
```

---

## 4) Huấn luyện & Quản lý Model

### 4.1 Train với MLflow tracking

```bash
python -m scripts.train_mlflow
```

Pipeline: train → cross-val 3 folds → log lên MLflow → so sánh F1 → nếu &gt;= best thì push `.tar.gz` lên MinIO → gọi Rasa reload API.

### 4.2 Deploy model hiện có lên MinIO

```bash
python -m scripts.upload_model_s3
```

### 4.3 Cơ chế load model trong container

- `endpoints.yml` trỏ `models.url` tới `MINIO_MODEL_URL` (VD: `http://minio:9000/chatbot-models/latest_model.tar.gz`)
- Rasa tự động download model vào `/tmp/` mỗi 10 giây (`wait_time_between_pulls: 10`)
- MinIO bucket cần set policy public read để Rasa GET được model qua HTTP (xem bước 1.6)

---

## 5) CI/CD với Jenkins

```bash
docker-compose up -d
```

Truy cập `http://localhost:8080`, pipeline đọc từ `Jenkinsfile`.

Pipeline gồm 4 stages:

1. **Data Pipeline** — `csv_to_rasa.py` chuyển chat logs thành NLU format
2. **Train Model** — `train_mlflow.py` train Rasa + log lên MLflow
3. **Human Approval** — gate thủ công, kiểm tra metrics trên MLflow
4. **Deploy** — `deploy_model.py` push model lên MinIO

---

## 6) Cấu hình môi trường (`.env`)

```bash
cp .env.example .env
```

### Database

| Var | Default |
| --- | --- |
| DB_HOST | localhost |
| DB_PORT | 5432 |
| DB_NAME | chatbot |
| DB_USER | chatbot_user |
| DB_PASSWORD | supersecret |

### Gemini (tuỳ chọn)

| Var | Default |
| --- | --- |
| GEMINI_API_KEY | *(để trống = tắt)* |
| GEMINI_MODEL | gemini-2.5-flash |

### Qdrant

| Var | Default |
| --- | --- |
| QDRANT_URL | http://localhost:6333 |
| QDRANT_COLLECTION | travel_destinations |
| QDRANT_TOUR_COLLECTION | travel_tours |

### MinIO (Model Registry)

| Var | Default |
| --- | --- |
| MINIO_URL | http://localhost:9000 |
| MINIO_ACCESS_KEY | admin |
| MINIO_SECRET_KEY | password123 |
| MINIO_BUCKET | chatbot-models |
| MINIO_MODEL_FILE | latest_model.tar.gz |

### MLflow

| Var | Default |
| --- | --- |
| MLFLOW_TRACKING_URI | http://localhost:5000 |
| MLFLOW_EXPERIMENT | Travel_Chatbot_Rasa |

### Rasa

| Var | Default |
| --- | --- |
| RASA_API_URL | http://localhost:5005 |
| RASA_ACTION_URL | http://localhost:5055/webhook |

### RabbitMQ

| Var | Default |
| --- | --- |
| RABBITMQ_HOST | localhost |
| RABBITMQ_PORT | 5672 |
| RABBITMQ_MGM_PORT | 15672 |

### Dashboard

| Var | Default |
| --- | --- |
| DASHBOARD_HOST | 0.0.0.0 |
| DASHBOARD_PORT | 5001 |

---

## 7) File & Script chính

| File | Chức năng |
| --- | --- |
| `Dockerfile` | Build image Rasa + Action Server |
| `scripts/docker-entrypoint.sh` | Entrypoint container: gen endpoints → start actions → start API |
| `scripts/generate_endpoints.py` | Thay `__PLACEHOLDER__` trong `endpoints.yml` bằng env vars |
| `scripts/start_all.sh` | Start local stack (Qdrant, RabbitMQ, actions, shell) daily |
| `scripts/stop_all.sh` | Dừng sạch local stack |
| `scripts/start_first_time.sh` | Setup từ đầu cho người mới clone |
| `scripts/train_mlflow.py` | Train + eval + auto-push MinIO |
| `scripts/deploy_model.py` | So sánh F1 → upload MinIO → reload Rasa |
| `scripts/upload_model_s3.py` | Upload model local lên MinIO |
| `scripts/sync_qdrant.py` | Sync embeddings destinations lên Qdrant |
| `scripts/sync_qdrant_tour.py` | Sync embeddings tours lên Qdrant |
| `scripts/check_qdrant.py` | Healthcheck Qdrant |
| `admin_dashboard/app.py` | Flask dashboard gán nhãn + retrain |
| `config/settings.py` | Tập trung tất cả env vars |
| `rasa_bot/endpoints.yml` | Template endpoints (có `__PLACEHOLDER__`) |
| `.env.example` | Mẫu cấu hình env |
| `.env.docker` | Env mẫu cho Docker test (dùng host.docker.internal) |
| `docker-compose.yml` | Jenkins container |
| `Dockerfile.jenkins` | Jenkins image + Python |

---

## 8) Khắc phục sự cố

| Lỗi | Cách fix |
| --- | --- |
| `pkg_resources` not found | `.venv/bin/pip install "setuptools<81"` |
| Qdrant collection not found | `.venv/bin/python -m scripts.sync_qdrant --recreate` |
| `relation "destinations" does not exist` | `.venv/bin/python -m database.init_db && .venv/bin/python -m database.importData` |
| Container không kết nối được DB | Dùng `--add-host host.docker.internal:host-gateway` hoặc trỏ DB_HOST đúng IP |
| Rasa không pull được model từ MinIO | Kiểm tra bucket policy public read, kiểm tra `MINIO_MODEL_URL` trong `endpoints.yml` |
| Action server bind error 5055 | Chỉ chạy 1 instance, dùng `scripts/stop_all.sh` dọn process cũ |

---

## 9) Ghi chú

- `setuptools<81` được pin trong `requirements.txt` để tránh lỗi `pkg_resources`.
- Qdrant dùng UUID deterministic khi upsert points để tránh lỗi point ID.
- `.env` và `rasa_bot/models/` được exclude trong `.dockerignore` → không lộ secret.
- Khi chạy Docker, model được load từ MinIO → không cần copy `models/` vào image.
- Dùng `.env.docker` thay `.env` khi chạy Rasa Server Docker container để test trong môi trường local (các host trỏ `host.docker.internal`).