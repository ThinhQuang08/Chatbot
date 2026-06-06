"""
deploy_model.py — Option A: Upload model artifact lên S3.
Jenkins sẽ tiếp tục build Docker image (model baked-in) sau khi script này thành công.
"""

import json
import glob
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ==========================================
# CONFIG — đọc từ env hoặc config/settings
# ==========================================
from config.settings import (
    AWS_DEFAULT_REGION,
    CHATBOT_S3_BUCKET,
    CHATBOT_S3_MODEL_KEY,
)

RASA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rasa_bot"))
RESULTS_DIR = os.path.join(RASA_DIR, "results")
MODEL_DIR = os.path.join(RASA_DIR, "models")
BEST_F1_RECORD_FILE = os.path.join(RASA_DIR, "best_f1_score.txt")

# boto3 S3 client — dùng AWS S3 thật (không endpoint_url)
s3_client = boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def get_current_f1():
    report_path = os.path.join(RESULTS_DIR, "intent_report.json")
    if not os.path.exists(report_path):
        return 0.0
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f).get("macro avg", {}).get("f1-score", 0.0)


def get_best_f1():
    if not os.path.exists(BEST_F1_RECORD_FILE):
        return 0.0
    with open(BEST_F1_RECORD_FILE, "r") as f:
        try:
            return float(f.read().strip())
        except ValueError:
            return 0.0


def upload_to_s3(local_file: str) -> bool:
    """Upload model file lên S3. Trả về True nếu thành công."""
    print(f"☁️  Uploading {os.path.basename(local_file)} → s3://{CHATBOT_S3_BUCKET}/{CHATBOT_S3_MODEL_KEY}")
    try:
        s3_client.upload_file(local_file, CHATBOT_S3_BUCKET, CHATBOT_S3_MODEL_KEY)
        print(f"✅ Upload thành công: s3://{CHATBOT_S3_BUCKET}/{CHATBOT_S3_MODEL_KEY}")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"❌ Upload S3 thất bại: {e}")
        return False


def run_cd_pipeline():
    print("-" * 60)
    print("🛡️  DEPLOY PIPELINE — Option A (S3 Upload)")
    print("-" * 60)

    current_f1 = get_current_f1()
    best_f1 = get_best_f1()

    print(f"📊 F1-Score vừa train : {current_f1:.4f}")
    print(f"🏆 F1-Score kỷ lục   : {best_f1:.4f}")

    list_of_models = glob.glob(os.path.join(MODEL_DIR, "*.tar.gz"))
    if not list_of_models:
        print("❌ Không tìm thấy file model nào trong rasa_bot/models/")
        raise FileNotFoundError("No model .tar.gz found")

    latest_model = max(list_of_models, key=os.path.getctime)
    print(f"📦 Model file: {os.path.basename(latest_model)}")

    # FAIL-SAFE: chỉ deploy nếu model mới tốt hơn
    if current_f1 < best_f1:
        print(f"\n❌ FAIL-SAFE: Model mới ({current_f1:.4f}) tệ hơn kỷ lục ({best_f1:.4f})")
        print("🗑️  Xóa model để tiết kiệm dung lượng...")
        os.remove(latest_model)
        print("🛑 Deploy bị hủy. S3 vẫn giữ bản model cũ an toàn.")
        raise SystemExit(1)

    print("\n✅ Model đạt chuẩn — tiến hành upload S3...")

    # Cập nhật record F1 tốt nhất
    with open(BEST_F1_RECORD_FILE, "w") as f:
        f.write(str(current_f1))

    # Upload lên S3
    success = upload_to_s3(latest_model)
    if not success:
        raise RuntimeError("S3 upload failed — aborting pipeline")

    print("\n🎉 S3 upload hoàn tất!")
    print("➡️  Jenkins sẽ tiếp tục: build Docker image → push DockerHub → update k8s-manifests image tag")


if __name__ == "__main__":
    run_cd_pipeline()