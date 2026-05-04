# data/seed_raw__data.py
# data fake
import psycopg2

DB_CONFIG = {
    "host": "localhost", # Đổi thành IP nếu chạy trên máy khác
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def seed_raw_data():
    try:
        print("[INFO] Đang kết nối tới Database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Dọn dẹp bảng cũ
        cur.execute("TRUNCATE TABLE ai_chat_analytics RESTART IDENTITY;")
        print("🧹 Đã dọn sạch bảng ai_chat_analytics. Chuẩn bị bơm data khủng...")

        # DỮ LIỆU MÔ PHỎNG THỰC TẾ (30 dòng)
        raw_data = [
            # 1. NHÓM NGOAN NGOÃN (Chuẩn ngữ pháp, Bot hiểu tốt)
            ("user_01", "tìm tour đà lạt 10 triệu", "search_travel", 0.95, "Đà Lạt", 10000000, 0),
            ("user_02", "cho tôi xem các tour đi Phú Quốc", "search_travel", 0.98, "Phú Quốc", None, 0),
            ("user_03", "giá tour nha trang 4 ngày 3 đêm là bao nhiêu", "ask_price", 0.96, "Nha Trang", None, 0),
            ("user_04", "mình muốn đặt phòng khách sạn ở sapa", "book_hotel", 0.94, "Sapa", None, 0),
            
            # 2. NHÓM TEENCODE & VIẾT TẮT HẠNG NẶNG (Cần Lexical Normalization)
            ("user_05", "ik sapa 8 củ có tour k bot", "search_travel", 0.72, "Sapa", 8000000, 0),
            ("user_06", "đlạt thág 10 có lạnh ko z", "ask_weather", 0.65, "Đà Lạt", None, 0),
            ("user_07", "mún đi hn chjll chjll 5tr way đầu", "search_travel", 0.58, "Hà Nội", 5000000, 0),
            ("user_08", "pù cúp có resot nào vjp ko", "search_travel", 0.45, "Phú Quốc", None, 0),
            ("user_09", "tour nt rẻ xĩu 2tr dc ko", "search_travel", 0.62, "Nha Trang", 2000000, 0),
            
            # 3. NHÓM KHÔNG DẤU & SAI CHÍNH TẢ
            ("user_10", "tim tour di da nang gia re", "search_travel", 0.88, "Đà Nẵng", None, 0),
            ("user_11", "gia tour sa pa nhiu tien", "ask_price", 0.85, "Sapa", None, 0),
            ("user_12", "đii phuu quốc chơiii", "search_travel", 0.75, "Phú Quốc", None, 0), # Lặp ký tự
            ("user_13", "khach sann o da lat dep wáááá", "book_hotel", 0.82, "Đà Lạt", None, 0),
            
            # 4. NHÓM HỎI TÀO LAO (Out of scope - Lạc đề)
            ("user_14", "thời tiết sài gòn nay sao", "out_of_scope", 0.99, None, None, 0),
            ("user_15", "bot biết giải toán không", "out_of_scope", 0.98, None, None, 0),
            ("user_16", "cách nấu bún bò huế ngon", "out_of_scope", 0.95, None, None, 0),
            ("user_17", "bạn tên là gì", "chitchat", 0.92, None, None, 0),
            ("user_18", "chửi thề dmm bot ngu", "toxicity", 0.99, None, None, 0),
            
            # 5. NHÓM NGÂN SÁCH DỊ THƯỜNG (Budget Drift)
            ("user_19", "cho tour nha trang 50 triệu", "search_travel", 0.96, "Nha Trang", 50000000, 0), # Quá cao
            ("user_20", "tour sapa 200k", "search_travel", 0.90, "Sapa", 200000, 0), # Quá thấp
            ("user_21", "có tour nào 1 tỷ đi đà lạt không", "search_travel", 0.85, "Đà Lạt", 1000000000, 0),
            
            # 6. NHÓM CÂU NGẮN & SPAM (Cần lọc rác)
            ("user_22", "a", "fallback", 0.20, None, None, 0), # Quá ngắn, bị lọc bỏ
            ("user_23", "ok", "affirm", 0.95, None, None, 0), # Quá ngắn
            ("user_24", "tour 500k", "search_travel", 0.90, "Sapa", 500000, 0), # Câu gốc
            ("user_25", "tour 500k", "search_travel", 0.90, "Sapa", 500000, 0), # Spam trùng lặp 1
            ("user_26", "tour 500k", "search_travel", 0.90, "Sapa", 500000, 0), # Spam trùng lặp 2
            
            # 7. NHÓM LÚ NÃO (Mập mờ, AI thiếu tự tin)
            ("user_27", "đi đâu khoảng 10tr mát mát mẻ mẻ", "search_travel", 0.55, None, 10000000, 0),
            ("user_28", "chỗ nào dạo này đang hot trend ta", "fallback", 0.40, None, None, 0),
            ("user_29", "muốn đi trốn nợ thì đi đâu", "out_of_scope", 0.60, None, None, 0),
            ("user_30", "cái đó đó, vé nhiêu", "ask_price", 0.35, None, None, 0)
        ]

        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget, user_feedback)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        # Thực thi bơm dữ liệu
        cur.executemany(sql, raw_data)
        conn.commit()
        
        print(f"✅ Đã bơm thành công {len(raw_data)} dòng data 'bẩn' vào PostgreSQL.")

    except Exception as e:
        print(f"❌ [ERROR DB]: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    seed_raw_data()