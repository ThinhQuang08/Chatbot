# Sử dụng Python 3.10 bản nhẹ để tối ưu dung lượng Image
FROM python:3.10-slim

# Khai báo thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các công cụ build cơ bản của hệ điều hành OS (Rất quan trọng cho Rasa)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào trước để tận dụng Docker Cache
COPY requirements.txt .

# Nâng cấp pip và cài đặt toàn bộ thư viện
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code cần thiết cho runtime (models được load từ MinIO)
COPY config/ config/
COPY services/ services/
COPY database/ database/
COPY utils/ utils/
COPY scripts/generate_endpoints.py scripts/
COPY scripts/docker-entrypoint.sh scripts/
COPY rasa_bot/ rasa_bot/

RUN chmod +x scripts/docker-entrypoint.sh

# Mở cổng action server (5055) và API server (5005)
EXPOSE 5005 5055

# Lệnh mặc định khi Container khởi chạy
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]