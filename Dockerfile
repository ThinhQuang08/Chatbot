FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Bước 1: cài torch CPU trước, chặn pip kéo CUDA wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Bước 2: cài phần còn lại, torch đã có sẵn nên pip không resolve lại
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5005