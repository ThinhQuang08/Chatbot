import pandas as pd
import os
import warnings

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import DB_HOST

# Thêm Dataset và DataDefinition từ cú pháp mới
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

    df_ref = pd.read_csv(REFERENCE_CSV_PATH)
    df_prod = pd.read_csv(PRODUCTION_CSV_PATH)

    if df_prod.empty:
        print("⚠️ Không có dữ liệu production")
        return

    # giữ các cột chung
    common_cols = list(
        set(df_ref.columns).intersection(
            set(df_prod.columns)
        )
    )

    df_ref = df_ref[common_cols]
    df_prod = df_prod[common_cols]

    print(
        f"[INFO] So sánh {len(df_prod)} dòng với {len(df_ref)} dòng"
    )

    # --- BẮT ĐẦU CẬP NHẬT THEO DOCS MỚI NHẤT ---
    # 1. Định nghĩa dữ liệu (Chỉ thêm tham số vào đây để Evidently hiểu đây là text)
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
    my_eval = report.run(
        reference_data=dataset_ref,
        current_data=dataset_prod
    )

    # 5. Xuất html TỪ BIẾN my_eval
    report_path = os.path.join(
        REPORT_OUTPUT_DIR,
        "drift_report_csv.html"
    )

    my_eval.save_html(report_path)

    print(f"✅ Đã lưu: {report_path}")

    # 6. Lấy dict TỪ BIẾN my_eval
    try:
        result = my_eval.as_dict()
    except AttributeError:
        # Fallback cho một số trường hợp version evidently xài .dict() thay vì .as_dict()
        result = my_eval.dict()

    try:
        # Lấy chính xác chỉ số drift của Evidently
        drift_share = (
            result["metrics"][0]
            ["result"]
            ["share_of_drifted_columns"]
        )
        is_drifted = result["metrics"][0]["result"]["dataset_drift"]
    except Exception as e:
        print(f"⚠️ Không đọc được JSON do khác phiên bản: {e}")
        drift_share = 0
        is_drifted = False

    print("-"*50)
    print(f"📊 Drift Share: {drift_share:.2%}")

    # Kết hợp cả điều kiện của Evidently phán xét hoặc tự set > 25%
    if is_drifted or drift_share > 0.25:
        print("🚨 BÁO ĐỘNG: ĐÃ PHÁT HIỆN DATA DRIFT!")
        return True

    print("🟢 Mô hình ổn định")
    return False


if __name__ == "__main__":
    check_drift_from_csv()
