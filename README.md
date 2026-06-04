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

## 1) Chạy Local (không Docker hóa Rasa)

### 1.1 Lần đầu tiên

```bash
bash scripts/start_first_time.sh
```

Script tự động: tạo venv → cài pip → khởi tạo DB → train model → push MinIO → chạy stack.

### 1.2 Chạy hàng ngày

```bash
bash scripts/start_all.sh
```

Tự động: kiểm tra PostgreSQL → khởi động Qdrant + RabbitMQ → sync vector → gen endpoints.yml → start action server → mở Rasa shell.

### 1.3 Dừng stack

```bash
bash scripts/stop_all.sh
# xoá luôn container Qdrant nếu muốn:
PURGE_QDRANT=true bash scripts/stop_all.sh
```

---

## 2) Chạy Docker (single image)

Build 1 image duy nhất chứa cả Rasa API + Action Server, model load từ MinIO.

### 2.1 Build image

```bash
docker build -t chatbot-rasa .
```

### 2.2 Chạy container

```bash
docker run -d \
  --name chatbot \
  -p 5005:5005 \
  -p 5055:5055 \
  --env-file .env \
  --add-host host.docker.internal:host-gateway \
  chatbot-rasa
```

Container sẽ: gen endpoints.yml → start action server (bg) → wait health → start Rasa API (fg cuối).

---

## 3) Huấn luyện & Quản lý Model

### 3.1 Train + Auto-deploy lên MinIO

```bash
python -m scripts.train_mlflow
```

Pipeline: train → cross-val → log lên MLflow → so sánh F1 → nếu &gt;= best thì push `.tar.gz` lên MinIO → gọi Rasa reload API.

### 3.2 Deploy model hiện có lên MinIO

```bash
python -m scripts.upload_model_s3
```

### 3.3 Cơ chế load model trong container

- `endpoints.yml` trỏ `models.url` tới `MINIO_MODEL_URL` (VD: `http://minio:9000/chatbot-models/latest_model.tar.gz`)
- Rasa tự động download model vào `/tmp/` mỗi 10 giây (`wait_time_between_pulls: 10`)
- MinIO bucket cần set policy public read để Rasa GET được model qua HTTP

---

## 4) Dashboard gán nhãn & Retrain

Flask app cho phép review dữ liệu Snorkel, gán nhãn lại, export NLU, và kick retrain.

```bash
python -m admin_dashboard.app
```

Truy cập: `http://localhost:5001`

---

## 5) CI/CD với Jenkins

Build Jenkins container có sẵn Python:

```bash
docker-compose up -d
```

Truy cập `http://localhost:8080`, pipeline đọc từ `Jenkinsfile`.

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
| `scripts/start_all.sh` | Start local stack (Qdrant, RabbitMQ, actions, shell) |
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
