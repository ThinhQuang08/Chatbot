import psycopg2
import random
import pandas as pd
import os

# 1. TẠO DỮ LIỆU CHUẨN (Để Bot tự học ngưỡng thống kê)
def create_fake_reference_data():
    os.makedirs('data', exist_ok=True)
    ref_data = []
    
    # Tạo 200 dòng data với mức giá phổ biến từ 8M - 12M
    # Phân phối này sẽ tạo ra Q1=9M, Q3=11M => IQR=2M
    # Lower Bound = Q1 - 1.5*IQR = 9M - 1.5*2M = 6.000.000 VNĐ
    budgets = [8000000, 9000000, 10000000, 11000000, 12000000]
    
    for _ in range(200):
        ref_data.append({
            'intent': 'search_travel',
            'destination': random.choice(['Đà Lạt', 'Nha Trang', 'Phú Quốc', 'Sapa']),
            'budget': random.choice(budgets)
        })
        
    df = pd.DataFrame(ref_data)
    df.to_csv('data/reference_logs.csv', index=False)
    print("✅ Đã tạo lại data/reference_logs.csv (Ngân sách chuẩn: 8M - 12M).")

# 2. BƠM CÁC CASE TEST VÀO DATABASE
DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def inject_test_cases():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Dọn sạch DB cũ để dễ quan sát (Test xong bạn có thể xóa dòng này)
        # cur.execute("DELETE FROM ai_chat_analytics;")
        
        test_cases = [
            # --- TRƯỜNG HỢP HỢP LỆ (SẼ BỊ LƯỚI LỌC BỎ QUA) ---
            ("user_good_1", "tìm tour đà lạt 10 triệu", "search_travel", 0.95, "Đà Lạt", 10000000),
            ("user_good_2", "đi sapa 8 củ", "search_travel", 0.98, "Sapa", 8000000),
            
            # --- TRƯỜNG HỢP RÁC (SẼ BỊ TÓM CỔ LƯU VÀO JSON) ---
            # 1. Bị bắt do Lỗi Intent (out_of_scope)
            ("user_bad_intent", "thời tiết nay sao", "out_of_scope", 0.99, None, None),
            
            # 2. Bị bắt do Lỗi Confidence (< 0.7)
            ("user_bad_conf", "đi đâu khoảng 10tr", "search_travel", 0.55, None, 10000000),
            
            # 3. Bị bắt do Budget Drift (Giá < 6 triệu theo AI tự tính)
            ("user_bad_budget_1", "cho tour nha trang 5 triệu", "search_travel", 0.96, "Nha Trang", 5000000),
            ("user_bad_budget_2", "tour 500k", "search_travel", 0.90, "Sapa", 500000)
        ]

        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        for case in test_cases:
            cur.execute(sql, case)

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Đã bơm {len(test_cases)} câu test đa dạng vào PostgreSQL.")
        
    except Exception as e:
        print(f"[ERROR DB] {e}")

if __name__ == "__main__":
    create_fake_reference_data()
    inject_test_cases()