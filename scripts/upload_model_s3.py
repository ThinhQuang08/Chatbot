import boto3
import os
import glob

from config.settings import (
    MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_BUCKET, MINIO_MODEL_FILE
)

s3_client = boto3.client('s3',
                         endpoint_url=MINIO_URL,
                         aws_access_key_id=MINIO_ACCESS_KEY,
                         aws_secret_access_key=MINIO_SECRET_KEY)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rasa_bot", "models"))

def upload_latest_model():
    print("[INFO] Đang tìm mô hình mới nhất trong thư mục local...")
    # Tìm file .tar.gz mới nhất
    list_of_files = glob.glob(os.path.join(MODEL_DIR, '*.tar.gz'))
    if not list_of_files:
        print("[ERROR] Không tìm thấy file mô hình nào trong thư mục models/")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    
    print(f"[INFO] Bắt đầu đẩy mô hình {os.path.basename(latest_file)} lên S3...")
    
    s3_client.upload_file(latest_file, MINIO_BUCKET, MINIO_MODEL_FILE)
    
    download_url = f"{MINIO_URL}/{MINIO_BUCKET}/{MINIO_MODEL_FILE}"
    print(f"[SUCCESS] Đã up lên Model Registry!")
    print(f"[URL] Đường dẫn kéo mô hình: {download_url}")

if __name__ == "__main__":
    upload_latest_model()