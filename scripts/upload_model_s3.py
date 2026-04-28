# test upload model
import boto3
import os
import glob

# Cấu hình kết nối tới MinIO
s3_client = boto3.client('s3',
                         endpoint_url='http://localhost:9000',
                         aws_access_key_id='admin',
                         aws_secret_access_key='password123')

BUCKET_NAME = 'chatbot-models'
MODEL_DIR = 'rasa_bot/models/'
# Tên file cố định trên S3 để Rasa luôn kéo đúng file này
TARGET_MODEL_NAME = 'latest_model.tar.gz'

def upload_latest_model():
    print("[INFO] Đang tìm mô hình mới nhất trong thư mục local...")
    # Tìm file .tar.gz mới nhất
    list_of_files = glob.glob(os.path.join(MODEL_DIR, '*.tar.gz'))
    if not list_of_files:
        print("[ERROR] Không tìm thấy file mô hình nào trong thư mục models/")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    
    print(f"[INFO] Bắt đầu đẩy mô hình {os.path.basename(latest_file)} lên S3...")
    
    # Upload lên MinIO và đổi tên thành latest_model.tar.gz
    s3_client.upload_file(latest_file, BUCKET_NAME, TARGET_MODEL_NAME)
    
    download_url = f"http://localhost:9000/{BUCKET_NAME}/{TARGET_MODEL_NAME}"
    print(f"[SUCCESS] Đã up lên Model Registry!")
    print(f"[URL] Đường dẫn kéo mô hình: {download_url}")

if __name__ == "__main__":
    upload_latest_model()