# scripts/train_mlflow.py
import mlflow
import yaml
import os
import subprocess
import glob
import time
import json

# Đường dẫn tĩnh dựa trên cấu trúc thư mục
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RASA_DIR = os.path.join(ROOT_DIR, "rasa_bot")
CONFIG_PATH = os.path.join(RASA_DIR, "config.yml")
MODEL_DIR = os.path.join(RASA_DIR, "models")
RESULTS_DIR = os.path.join(RASA_DIR, "results")

def train_and_evaluate_mlflow():
    if mlflow.active_run():
        mlflow.end_run()
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Travel_Chatbot_Rasa")

    # Tên mô hình
    with mlflow.start_run(run_name=f"Train_Eval_{int(time.time())}"):
        
        # ==========================================
        # 1. ĐỌC THÔNG SỐ CẤU HÌNH (PARAMETERS)
        # ==========================================
        print("📊 Đang đọc thông số từ config.yml...")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            mlflow.log_param("language", config.get("language"))
            mlflow.log_param("num_policies", len(config.get("policies", [])))
            for component in config.get("pipeline", []):
                if component.get("name") == "DIETClassifier":
                    mlflow.log_param("DIET_epochs", component.get("epochs", "default"))
                    mlflow.log_param("DIET_learning_rate", component.get("learning_rate", "default"))

        # ==========================================
        # 2. HUẤN LUYỆN MODEL (TRAIN)
        # ==========================================
        print("🚀 Bắt đầu huấn luyện Rasa Model...")
        start_time = time.time()
        train_result = subprocess.run(["rasa", "train"], cwd=RASA_DIR, capture_output=True, text=True)
        training_time = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", training_time)

        if train_result.returncode != 0:
            print("❌ Lỗi trong quá trình huấn luyện:\n", train_result.stderr)
            with open("error_log.txt", "w") as err_file:
                err_file.write(train_result.stderr)
            mlflow.log_artifact("error_log.txt")
            return

        print("✅ Huấn luyện thành công!")
        list_of_models = glob.glob(os.path.join(MODEL_DIR, '*.tar.gz'))
        if not list_of_models:
            print("⚠️ Không tìm thấy file model.")
            return
            
        latest_model = max(list_of_models, key=os.path.getctime)
        print(f"📦 Đã đóng gói model: {os.path.basename(latest_model)}")

        # ==========================================
        # 3. ĐÁNH GIÁ MODEL (EVALUATE/TEST NLU)
        # ==========================================
        print("🧪 Đang làm bài kiểm tra Cross-Validation (Xác thực chéo 3 vòng)...")
        print("⏳ Quá trình này sẽ hơi lâu một chút để đảm bảo điểm số là điểm THẬT 100%.")
        
        # test_result = subprocess.run(
        #     [
        #         "rasa", "test", "nlu", 
        #         "--nlu", "train_test_split/test_data.yml", 
        #         "--model", latest_model
        #     ], 
        #     cwd=RASA_DIR, 
        #     capture_output=True, 
        #     text=True
        # )
        # khi >75% Dùng cross-validation tự động phân chia data để chống "Lộ đề thi" (Data Leakage)
        test_result = subprocess.run(
            ["rasa", "test", "nlu", "--cross-validation", "--folds", "3"], 
            cwd=RASA_DIR, 
            capture_output=True, 
            text=True
        )



        # ==========================================
        # 4. PHÂN TÍCH ĐIỂM VÀ ĐẨY LÊN MLFLOW
        # ==========================================
        report_path = os.path.join(RESULTS_DIR, "intent_report.json")
        if os.path.exists(report_path):
            print("📈 Đang phân tích bảng điểm bài kiểm tra...")
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                
                # Trích xuất các chỉ số cực kỳ quan trọng để show lên MLflow
                accuracy = report.get("accuracy", 0.0)
                mlflow.log_metric("nlu_accuracy", accuracy)
                
                # Macro Avg (Điểm trung bình chia đều cho mọi intent)
                macro_avg = report.get("macro avg", {})
                mlflow.log_metric("nlu_f1_score", macro_avg.get("f1-score", 0.0))
                mlflow.log_metric("nlu_precision", macro_avg.get("precision", 0.0))
                mlflow.log_metric("nlu_recall", macro_avg.get("recall", 0.0))
        else:
            print("Không tìm thấy file intent_report.json. Bot có thể chưa có đủ data test.")

        # Đẩy toàn bộ "Tài sản" lên MLflow
        print("Đang đồng bộ Artifacts lên MLflow...")
        mlflow.log_artifact(latest_model, artifact_path="rasa_model")
        mlflow.log_artifact(CONFIG_PATH, artifact_path="configs")
        
        # Đẩy nguyên thư mục results (chứa báo cáo lỗi, biểu đồ ma trận nhầm lẫn)
        if os.path.exists(RESULTS_DIR):
            mlflow.log_artifacts(RESULTS_DIR, artifact_path="evaluation_results")
        
        print("Tích hợp MLOps hoàn tất! Mở http://localhost:5000 để chiêm ngưỡng thành quả.")

if __name__ == "__main__":
    train_and_evaluate_mlflow()