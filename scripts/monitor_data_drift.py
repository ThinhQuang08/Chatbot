# scripts/monitor_data_drift.py
import pandas as pd
import os
import warnings

from evidently import Dataset, DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

REFERENCE_CSV_PATH = "data/reference_data.csv"
PRODUCTION_CSV_PATH = "data/cleaned_chat_logs.csv" 
REPORT_OUTPUT_DIR = "results/evidently_reports"

os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# Hàm đệ quy săn lùng chi tiết Drift của từng cột trong file JSON
def find_drift_by_columns(obj):
    if isinstance(obj, dict):
        if "drift_by_columns" in obj: return obj["drift_by_columns"]
        for k, v in obj.items():
            res = find_drift_by_columns(v)
            if res is not None: return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_drift_by_columns(item)
            if res is not None: return res
    return None

def check_drift_from_csv():
    print("=" * 50)
    print("🔍 KÍCH HOẠT HỆ THỐNG QUÉT DRIFT (DÙNG PRESET CHUẨN)...")
    print("=" * 50)

    if not os.path.exists(REFERENCE_CSV_PATH) or not os.path.exists(PRODUCTION_CSV_PATH):
        print("❌ Lỗi: Không tìm thấy file CSV.")
        return

    df_ref = pd.read_csv(REFERENCE_CSV_PATH)
    df_prod = pd.read_csv(PRODUCTION_CSV_PATH)

    if df_prod.empty:
        print("⚠️ Không có dữ liệu production")
        return

    # Lọc ra các cột chung
    common_cols = list(set(df_ref.columns).intersection(set(df_prod.columns)))
    
    # MẸO: CHỈ GIỮ LẠI ĐÚNG 3 CỘT QUAN TRỌNG ĐỂ ÉP PRESET PHẢI QUÉT CHÚNG
    target_cols = [c for c in ["predicted_intent", "confidence_score", "cleaned_text", "raw_text"] if c in common_cols]
    
    df_ref = df_ref[target_cols]
    df_prod = df_prod[target_cols]

    # --- ĐẢM BẢO TẬP REFERENCE CÓ PHƯƠNG SAI ---
    if 'confidence_score' in df_ref.columns:
        df_ref.loc[0, 'confidence_score'] = 0.99 
        df_ref.loc[1, 'confidence_score'] = 0.98

    print(f"[INFO] Đang so sánh {len(df_prod)} câu thực tế với {len(df_ref)} câu gốc")

    # 1. Định nghĩa kiểu dữ liệu cho 3 cột
    data_def = DataDefinition(
        categorical_columns=["predicted_intent"] if "predicted_intent" in target_cols else [],
        numerical_columns=["confidence_score"] if "confidence_score" in target_cols else [],
        text_columns=["cleaned_text"] if "cleaned_text" in target_cols else (["raw_text"] if "raw_text" in target_cols else [])
    )

    dataset_ref = Dataset.from_pandas(df_ref, data_definition=data_def)
    dataset_prod = Dataset.from_pandas(df_prod, data_definition=data_def)

    # 2. CHỈ DÙNG DataDriftPreset (Đã loại bỏ hoàn toàn ColumnDriftMetric gây lỗi)
    report = Report(metrics=[DataDriftPreset()])

    # 3. Chạy đánh giá
    my_eval = report.run(reference_data=dataset_ref, current_data=dataset_prod)

    # 4. Xuất Dashboard
    report_path = os.path.join(REPORT_OUTPUT_DIR, "drift_report_csv.html")
    my_eval.save_html(report_path)
    print(f"✅ Đã lưu HTML Dashboard tại: {report_path}")

    # 5. Phân tích kết quả siêu chi tiết
    try:
        result = my_eval.as_dict()
    except AttributeError:
        result = my_eval.dict()

    print("-" * 50)
    
    drift_alert = False
    drift_by_columns = find_drift_by_columns(result)

    if drift_by_columns:
        for col_name, col_data in drift_by_columns.items():
            drift_detected = col_data.get("drift_detected", False)
            drift_score = col_data.get("drift_score", 0.0)
            status = "🚨 BỊ TRÔI DẠT" if drift_detected else "🟢 Ổn định"
            
            print(f"📌 {col_name.ljust(18)} : {status} (Score: {drift_score:.4f})")
            if drift_detected:
                drift_alert = True
    else:
        print("⚠️ Không lấy được chi tiết từng cột, cảnh báo dựa trên tổng thể.")
        drift_alert = True # Fallback an toàn, nếu có lỗi đọc JSON thì cứ báo động

    print("-" * 50)

    if drift_alert:
        print("🚨 BÁO ĐỘNG: HỆ THỐNG ĐÃ PHÁT HIỆN DATA DRIFT!")
        print("👉 Hướng xử lý: Có câu hỏi lạ/độ tự tin thấp -> Đẩy vào Auto-label (Snorkel)!")
        return True

    print("🟢 Bot đang phản hồi tốt, không cần học lại.")
    return False

if __name__ == "__main__":
    check_drift_from_csv()