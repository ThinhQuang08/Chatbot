# tạo data drift ảo
import psycopg2
import random

# Cấu hình DB giống hệt con Worker
DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def inject_drift_data():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print("[INFO] Đang bơm 100 dòng Data Drift vào Database...")

        for i in range(100):
            session_id = f"fake_drift_user_{i}"
            
            # Mặc định là tìm tour giá cực bèo (Budget Drift)
            raw_text = "tìm tour giá siêu rẻ sinh viên"
            intent = "search_travel"
            confidence = random.uniform(0.85, 0.99)
            destination = random.choice(["Vũng Tàu", "Đà Lạt", "Sapa"])
            parsed_budget = random.randint(300000, 900000) # Chỉ từ 300k - 900k

            # Xác suất 30% khách hỏi tào lao (Intent Drift)
            if random.random() < 0.3:
                intent = "out_of_scope"
                raw_text = "nay trời có mưa không bot"
                destination = None
                parsed_budget = None

            sql = """
                INSERT INTO ai_chat_analytics 
                (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(sql, (session_id, raw_text, intent, confidence, destination, parsed_budget))

        conn.commit()
        cur.close()
        conn.close()
        print("[SUCCESS] Bơm dữ liệu thành công! Hãy chạy file monitor_drift.py để xem kết quả.")
        
    except Exception as e:
        print(f"[ERROR] Lỗi nạp đạn: {e}")

if __name__ == "__main__":
    inject_drift_data()