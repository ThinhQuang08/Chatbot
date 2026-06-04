import pandas as pd
import os
import warnings
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Sử dụng chính xác cú pháp import từ file của sếp
from evidently import Dataset, DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCE_CSV_PATH = os.getenv("REFERENCE_CSV_PATH", os.path.join(ROOT_DIR, "data", "reference_data.csv"))
PRODUCTION_CSV_PATH = os.getenv("PRODUCTION_CSV_PATH", os.path.join(ROOT_DIR, "data", "cleaned_chat_logs.csv"))
REPORT_OUTPUT_DIR = os.getenv("EVIDENTLY_REPORT_DIR", os.path.join(ROOT_DIR, "results", "evidently_reports"))

os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


def check_drift_from_csv():

    print("=" * 50)
    print("🔍 KÍCH HOẠT HỆ THỐNG QUÉT DRIFT TRÊN CSV...")
    print("=" * 50)

    try:
        df_ref = pd.read_csv(REFERENCE_CSV_PATH)
        df_prod = pd.read_csv(PRODUCTION_CSV_PATH)
    except FileNotFoundError as e:
        print(f"⚠️ Lỗi: Không tìm thấy file dữ liệu - {e}")
        return False

    if df_prod.empty:
        print("⚠️ Không có dữ liệu production")
        return False

    # Giữ các cột chung để so sánh
    common_cols = list(set(df_ref.columns).intersection(set(df_prod.columns)))
    df_ref = df_ref[common_cols]
    df_prod = df_prod[common_cols]

    print(f"[INFO] So sánh {len(df_prod)} dòng production với {len(df_ref)} dòng reference")

    # 1. Định nghĩa dữ liệu theo đúng cú pháp của sếp
    data_def = DataDefinition(
        categorical_columns=["predicted_intent"] if "predicted_intent" in common_cols else [],
        text_columns=["cleaned_text"] if "cleaned_text" in common_cols else []
    )

    # 2. Bọc Pandas DataFrame lại thành Dataset của Evidently
    dataset_ref = Dataset.from_pandas(df_ref, data_definition=data_def)
    dataset_prod = Dataset.from_pandas(df_prod, data_definition=data_def)

    # 3. Tạo cấu hình Report
    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    # 4. Hứng kết quả chạy vào biến my_eval
    print("⏳ Đang tính toán độ lệch dữ liệu...")
    my_eval = report.run(
        reference_data=dataset_ref,
        current_data=dataset_prod
    )

    # 5. Xuất HTML từ biến my_eval
    report_path = os.path.join(REPORT_OUTPUT_DIR, "drift_report_csv.html")
    my_eval.save_html(report_path)
    print(f"✅ Đã lưu HTML Report tại: {report_path}")

    # 6. Lấy dict từ biến my_eval
    try:
        result = my_eval.as_dict()
    except AttributeError:
        result = my_eval.dict()

    # 7. Trích xuất chỉ số Drift (Chuẩn Evidently v2 API)
    drift_share = 0
    is_drifted = False

    try:
        # Xử lý linh hoạt việc result trả về là Dict chứa "metrics" hay là List
        metrics_list = result.get("metrics", result) if isinstance(result, dict) else result
        
        for metric_data in metrics_list:
            # Tìm trong config type
            config_type = metric_data.get("config", {}).get("type", "")
            
            if "DriftedColumnsCount" in config_type:
                # Lấy tỷ lệ drift
                value_data = metric_data.get("value", {})
                drift_share = value_data.get("share", 0)
                
                # So sánh với ngưỡng (threshold) của Evidently để xem có alert không
                threshold = metric_data.get("config", {}).get("drift_share", 0.5) # Thường mặc định là 0.5 (50%)
                is_drifted = drift_share >= threshold
                break
    except Exception as e:
        print(f"⚠️ Lỗi trích xuất JSON: {e}")

    print("-" * 50)
    print(f"📊 Drift Share (Tỷ lệ cột lệch): {drift_share:.2%}")

    # Cảnh báo nếu Drift vượt ngưỡng của Evidently hoặc tự set > 25%
    if is_drifted or drift_share > 0.25:
        print("🚨 BÁO ĐỘNG ĐỎ: ĐÃ PHÁT HIỆN DATA DRIFT!")
        return True

    print("🟢 Mô hình ổn định")
    return False

if __name__ == "__main__":
    check_drift_from_csv()