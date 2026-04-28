import pandas as pd
import os
import time
import schedule
from datetime import datetime
import psycopg2
import warnings

from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings('ignore')

DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_current_chat_logs():
    print("[INFO] Đang kéo dữ liệu 7 ngày qua từ PostgreSQL...")
    conn = get_connection()
    query = """
        SELECT 
            session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget
        FROM ai_chat_analytics 
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        AND predicted_intent IS NOT NULL;
    """
    try:
        current_data = pd.read_sql_query(query, conn)
        current_data = current_data.rename(columns={
            'predicted_intent': 'intent',
            'parsed_budget': 'budget'
        })
        return current_data
    except Exception as e:
        print(f"[ERROR DB] {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def extract_anomalies(current_df, reference_df=None):
    """Lọc các câu chat bất thường bằng Cấu hình động và Thống kê IQR"""
    
    # 1. TÁCH BIỆT CẤU HÌNH (Sau này có thể đọc từ file .env hoặc config.yaml)
    CONFIG = {
        "min_confidence": 0.7,
        "anomaly_intents": ['out_of_scope', 'fallback']
    }

    # Bắt lỗi theo Rule-based (Dựa trên cấu hình)
    cond_intent = current_df['intent'].isin(CONFIG["anomaly_intents"])
    cond_conf = current_df['confidence_score'] < CONFIG["min_confidence"]

    # Khởi tạo điều kiện budget mặc định là False (không bắt gì cả)
    cond_budget = pd.Series(False, index=current_df.index)

    # 2. BẮT LỖI THEO THỐNG KÊ ĐỘNG (Tự động tính ngưỡng từ Reference Data)
    if reference_df is not None and 'budget' in reference_df.columns:
        # Loại bỏ các giá trị null hoặc bằng 0 trong tập chuẩn trước khi tính
        valid_ref_budgets = reference_df[reference_df['budget'] > 0]['budget']
        
        if not valid_ref_budgets.empty:
            Q1 = valid_ref_budgets.quantile(0.25)
            Q3 = valid_ref_budgets.quantile(0.75)
            IQR = Q3 - Q1
            
            # Tính giới hạn dưới an toàn (Nếu < 0 thì set bằng 0)
            lower_bound = max(0, Q1 - 1.5 * IQR)
            
            # Bắt các budget hiện tại thấp hơn ngưỡng tự học này
            cond_budget = (current_df['budget'] < lower_bound) & (current_df['budget'] > 0)
            
            print(f"[STATISTICS] Hệ thống tự học ngưỡng Budget bình thường thấp nhất: {lower_bound:,.0f} VND")

    # Gom tất cả rác lại
    anomalies = current_df[cond_intent | cond_conf | cond_budget]
    
    return anomalies[['session_id', 'raw_text', 'intent', 'confidence_score', 'budget']]

def run_automated_monitoring():
    print("="*50)
    print(f"[⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] KÍCH HOẠT QUÉT DATA DRIFT")
    print("="*50)
    
    # 1. Kéo data
    ref_path = "data/reference_logs.csv"
    reference_data = pd.read_csv(ref_path)
    current_data = fetch_current_chat_logs()
    
    if current_data.empty:
        print("[WARN] Không có dữ liệu mới. Bỏ qua.")
        return

    cols_to_monitor = ['intent', 'destination', 'budget'] 
    ref_for_evidently = reference_data[cols_to_monitor]
    cur_for_evidently = current_data[cols_to_monitor]

    # 2. Sinh báo cáo HTML (Dành cho việc show Dashboard)
    print("[INFO] Đang phân tích biểu đồ phân phối...")
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_for_evidently, current_data=cur_for_evidently)
    
    html_path = "evidently_drift_report.html"
    my_eval.save_html(html_path)
    print(f"📊 Đã xuất báo cáo Dashboard thành công: {html_path}")

    # 3. ĐI THẲNG VÀO TRÍCH XUẤT DATA RÁC (Bỏ qua bóc tách Dictionary)
    anomalies_df = extract_anomalies(current_data, reference_data)
    
    if not anomalies_df.empty:
        print(f"🚨 [CẢNH BÁO] Phát hiện có luồng dữ liệu rác/bất thường!")
        export_path = "data/drifted_anomalies.json"
        anomalies_df.to_json(export_path, orient='records', force_ascii=False, indent=4)
        print(f"📥 Đã gom {len(anomalies_df)} câu chat gửi vào kho chờ LLM xử lý: {export_path}")
    else:
        print("✅ Dữ liệu ổn định. Bot đang trả lời tốt, không cần Retrain.")
        
    print("-" * 50 + "\n")

if __name__ == "__main__":
    print("[SYSTEM] Khởi động nhân viên trực ban MLOps...")
    
    # Chạy mồi lần 1 ngay lập tức để lấy file JSON
    run_automated_monitoring()
    
    # Cấu hình treo Terminal tự động chạy vào 2h sáng (hoặc test bằng schedule.every(10).seconds.do...)
    schedule.every().day.at("02:00").do(run_automated_monitoring)
    
    while True:
        schedule.run_pending()
        time.sleep(1)