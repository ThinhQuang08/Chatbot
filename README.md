# Chatbot Du Lịch (Rasa + PostgreSQL + Qdrant)

Dự án này là chatbot tư vấn du lịch tiếng Việt, dùng:

- Rasa cho NLU + hội thoại
- PostgreSQL cho dữ liệu điểm đến
- Qdrant cho semantic rerank (vector search)
- (Tuỳ chọn) Gemini để sinh câu trả lời tự nhiên dựa trên kết quả truy xuất

## 1) Yêu cầu máy cài đặt

- Linux/macOS (khuyến nghị Ubuntu 22.04+)
- Python 3.10
- Docker
- PostgreSQL (đã chạy và có database)

### Nếu chưa có PostgreSQL: chạy nhanh bằng Docker

Bạn có thể dựng PostgreSQL bằng 1 lệnh (đúng luôn với `.env.example`):

```bash
docker run -d \
	--name chatbot-postgres \
	-e POSTGRES_USER=chatbot_user \
	-e POSTGRES_PASSWORD=supersecret \
	-e POSTGRES_DB=chatbot \
	-p 5432:5432 \
	-v chatbot_pg_data:/var/lib/postgresql/data \
	postgres:15
```

Kiểm tra container chạy:

```bash
docker ps --filter name=chatbot-postgres
```

Kiểm tra kết nối DB:

```bash
docker exec -it chatbot-postgres psql -U chatbot_user -d chatbot -c "SELECT 1;"
```

Nếu cần dừng/xoá PostgreSQL container:

```bash
docker stop chatbot-postgres
docker rm chatbot-postgres
# (tuỳ chọn) xoá luôn dữ liệu volume:
docker volume rm chatbot_pg_data
```

## 2) Cách chạy dự án (One-command)

### 2.1) Dành cho người mới cài lần đầu (Vừa git clone)

Khi vừa tải dự án về, chạy duy nhất script khởi tạo tự động toàn bộ môi trường (PostgreSQL, Qdrant, Dependencies, Nạp dữ liệu, Train Rasa Model):

```bash
cd Chatbot
bash scripts/start_first_time.sh
```

### 2.2) Chạy hàng ngày (Daily Run)

Khi đã setup dữ liệu và mô hình ở lần đầu thành công, các lần khởi động sau bạn chỉ cần:

```bash
cd Chatbot
bash scripts/start_all.sh
```

Lưu ý: lệnh chạy hàng ngày giúp thay đổi (code, webhook) có hiệu lực rất nhanh vì bỏ qua cài PIP và tái nạp Database, tuy nhiên nó sẽ không phục hồi PostgreSQL nếu container đã bị tắt. Đảm bảo `chatbot-postgres` vẫn đang chạy.

Để dừng sạch toàn bộ stack (actions + rasa process còn sót + Qdrant container):

```bash
bash scripts/stop_all.sh
```

Script `start_all.sh` sẽ tự động:

1. Tạo `.venv` nếu chưa có
2. Cài dependencies từ `requirements.txt`
3. Tạo `.env` từ `.env.example` nếu thiếu
4. Kiểm tra PostgreSQL và tự import dữ liệu nếu bảng `destinations` chưa tồn tại
5. Khởi động Qdrant bằng Docker (có volume để giữ dữ liệu)
6. Sync embeddings vào Qdrant
7. Chạy `rasa run actions`
8. Mở `rasa shell` để test

Script `stop_all.sh` sẽ tự động:

1. Dừng action server theo PID runtime
2. Quét và dừng các process Rasa còn sót trong thư mục project
3. Dừng Qdrant container
4. Dọn file runtime state để tránh ghost process lần chạy sau

## 3) Cấu hình môi trường

### 3.1 Tạo file .env

```bash
cp .env.example .env
```

Giá trị cần chỉnh tối thiểu trong `.env`:

- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `GEMINI_API_KEY` (nếu muốn bật trả lời bằng Gemini)

Model semantic mặc định đang để:

- `SEMANTIC_MODEL_NAME=keepitreal/vietnamese-sbert`

## 4) Chạy thủ công (nếu không dùng one-command)

```bash
cd Chatbot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # nếu chưa có

# nếu chưa có PostgreSQL thì chạy container PostgreSQL trước
docker run -d --name chatbot-postgres -e POSTGRES_USER=chatbot_user -e POSTGRES_PASSWORD=supersecret -e POSTGRES_DB=chatbot -p 5432:5432 -v chatbot_pg_data:/var/lib/postgresql/data postgres:15

# import dữ liệu điểm đến vào PostgreSQL
.venv/bin/python -m database.import_data

docker run -d --name chatbot-qdrant -p 6333:6333 -p 6334:6334 -v chatbot_qdrant_data:/qdrant/storage qdrant/qdrant

.venv/bin/python -m scripts.sync_qdrant --recreate

cd rasa_bot
../.venv/bin/rasa run actions
# terminal khác:
../.venv/bin/rasa shell --model models/level45-level5-v2.tar.gz
```

## 5) Ghi chú quan trọng

- Để tránh lỗi `pkg_resources`, đã pin `setuptools<81` trong `requirements.txt`.
- Để tránh lỗi Qdrant point ID, hệ thống dùng UUID deterministic khi upsert points.
- Nếu thay model semantic, nên sync lại:

```bash
.venv/bin/python -m scripts.sync_qdrant --recreate
```

## 6) Script và file chính

- Script one-command: [scripts/start_all.sh](scripts/start_all.sh)
- Script dừng sạch stack: [scripts/stop_all.sh](scripts/stop_all.sh)
- Sync vector store: [scripts/sync_qdrant.py](scripts/sync_qdrant.py)
- Healthcheck Qdrant: [scripts/check_qdrant.py](scripts/check_qdrant.py)
- Cấu hình env mẫu: [.env.example](.env.example)
- Dependency pin: [requirements.txt](requirements.txt)

### Tuỳ chọn dọn sâu Qdrant container

Nếu muốn dừng **và xoá container** Qdrant:

```bash
PURGE_QDRANT=true bash scripts/stop_all.sh
```

## 7) Khắc phục sự cố nhanh

### Lỗi `ModuleNotFoundError: pkg_resources`

Chạy lại:

```bash
.venv/bin/pip install "setuptools<81"
```

### Qdrant báo collection chưa tồn tại

Chạy lại sync:

```bash
.venv/bin/python -m scripts.sync_qdrant --recreate
```

### Lỗi `relation "destinations" does not exist`

Chạy import dữ liệu PostgreSQL:

```bash
.venv/bin/python -m database.import_data
```

hoặc chạy lại one-command script (script đã tự bootstrap bảng nếu thiếu):

```bash
bash scripts/start_all.sh
```

### Chạy bằng sai Python interpreter

Luôn dùng binary trong `.venv`, ví dụ:

- `.venv/bin/python`
- `.venv/bin/rasa`
