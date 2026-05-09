import json
import glob
import os
import requests
import boto3

# ==========================================
# CẤU HÌNH HỆ THỐNG & MINIO
# ==========================================
RASA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rasa_bot"))
RESULTS_DIR = os.path.join(RASA_DIR, "results")
MODEL_DIR = os.path.join(RASA_DIR, "models")
BEST_F1_RECORD_FILE = os.path.join(RASA_DIR, "best_f1_score.txt")

BUCKET_NAME = 'chatbot-models'
TARGET_MODEL_NAME = 'latest_model.tar.gz'
MINIO_URL = 'http://localhost:9000'

s3_client = boto3.client('s3',
                         endpoint_url=MINIO_URL,
                         aws_access_key_id='admin',
                         aws_secret_access_key='password123')

def get_current_f1():
    report_path = os.path.join(RESULTS_DIR, "intent_report.json")
    if not os.path.exists(report_path): return 0.0
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f).get("macro avg", {}).get("f1-score", 0.0)

def get_best_f1():
    if not os.path.exists(BEST_F1_RECORD_FILE): return 0.0
    with open(BEST_F1_RECORD_FILE, "r") as f:
        try: return float(f.read().strip())
        except ValueError: return 0.0

def upload_to_minio(latest_file):
    print(f"☁️ Đang đẩy mô hình {os.path.basename(latest_file)} lên MinIO...")
    try:
        s3_client.upload_file(latest_file, BUCKET_NAME, TARGET_MODEL_NAME)
        download_url = f"{MINIO_URL}/{BUCKET_NAME}/{TARGET_MODEL_NAME}"
        print(f"✅ Đã up lên Model Registry: {download_url}")
        return download_url
    except Exception as e:
        print(f"❌ Lỗi đẩy MinIO: {e}")
        return None

def trigger_rasa_reload(model_url):
    print("🔔 Đang gọi Webhook API ép Rasa nạp lại mô hình mới từ MinIO...")
    # Gọi API của Rasa để thay thế mô hình đang chạy trong bộ nhớ
    payload = {
        "model_server": {
            "url": model_url
        }
    }
    try:
        # Rasa mặc định mở API ở cổng 5005
        response = requests.put("http://localhost:5005/model", json=payload)
        if response.status_code == 204:
            print("🚀 THÀNH CÔNG: Rasa đã nạp mô hình mới nóng hổi. Bot đã thông minh hơn!")
        else:
            print(f"⚠️ Rasa phản hồi lạ: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Không gọi được Rasa. Hãy chắc chắn sếp đã chạy Rasa với cờ --enable-api")

def run_cd_pipeline():
    print("-" * 50)
    print("🛡️ KÍCH HOẠT HỆ THỐNG GÁC CỔNG VÀ DEPLOY (MINIO)...")
    print("-" * 50)
    
    current_f1 = get_current_f1()
    best_f1 = get_best_f1()
    
    print(f"📊 F1-Score Mô hình vừa học: {current_f1:.4f}")
    print(f"🏆 F1-Score Kỷ lục cũ      : {best_f1:.4f}")
    
    list_of_models = glob.glob(os.path.join(MODEL_DIR, '*.tar.gz'))
    if not list_of_models:
        print("❌ Không tìm thấy file model nào.")
        return
        
    latest_model = max(list_of_models, key=os.path.getctime)
    
    # 1. KIỂM ĐỊNH (FAIL-SAFE)
    if current_f1 < best_f1:
        print("\n❌ FAIL-SAFE KÍCH HOẠT: Mô hình mới TỆ HƠN mô hình cũ!")
        print(f"🗑️ Đang xóa mô hình {os.path.basename(latest_model)} để bảo vệ chất lượng...")
        os.remove(latest_model)
        print("🛑 Đã hủy quy trình Deploy. MinIO vẫn giữ bản model cũ an toàn.")
        return

    print("\n✅ PASS: Mô hình đạt chuẩn! Tiến hành Deploy...")
    with open(BEST_F1_RECORD_FILE, "w") as f:
        f.write(str(current_f1))
        
    # 2. PUSH MODEL LÊN MINIO
    model_url = upload_to_minio(latest_model)
    
    # 3. KÍCH HOẠT RASA RELOAD
    if model_url:
        trigger_rasa_reload(model_url)

if __name__ == "__main__":
    run_cd_pipeline()