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

# Copy toàn bộ code (models, actions, data, config...) vào container
COPY . .

# Mở cổng 5005 để giao tiếp API với thế giới bên ngoài
EXPOSE 5005

# Lệnh mặc định khi Container khởi chạy
# (Bật API, mở CORS cho web client và trỏ tới file endpoints)
# CMD ["rasa", "run", "--enable-api", "--cors", "*", "--endpoints", "endpoints.yml"]