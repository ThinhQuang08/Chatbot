import psycopg2
import random

DB_CONFIG = {
    "host": "localhost",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def generate_massive_data(num_samples=1500):
    destinations = ["đà lạt", "sapa", "nha trang", "phú quốc", "đà nẵng", "hà nội", "pù cúp", "đlạt", "nt", "sg"]
    price_keywords = ["giá nhiêu", "mấy tiền", "chi phí", "bao nhiêu", "giá rẻ", "mấy củ", "khoảng mấy tr", "giá sao"]
    book_keywords = ["đặt phòng", "book tour", "thuê ks", "đặt vé", "tìm chỗ ở", "đặt ks", "book vé"]
    out_of_scope = ["thời tiết nay sao", "nấu ăn", "chửi thề", "giải bài tập", "bot ngu", "hôm nay ăn gì", "bạn tên gì", "buồn ngủ quá"]

    raw_data = []
    
    print(f"[INFO] Đang trộn từ vựng để sinh ra {num_samples} câu chat...")
    for i in range(num_samples):
        rand = random.random()
        # 40% dữ liệu: Tìm tour / Hỏi địa điểm
        if rand < 0.40: 
            dest = random.choice(destinations)
            prefix = random.choice(["tìm tour", "muốn đi", "có tour nào đi", "review du lịch", "ik"])
            text = f"{prefix} {dest}"
            
        # 30% dữ liệu: Hỏi giá
        elif rand < 0.70: 
            dest = random.choice(destinations)
            price = random.choice(price_keywords)
            text = f"tour {dest} {price}" if random.random() > 0.5 else f"{price} cho tour {dest}"
            
        # 20% dữ liệu: Đặt phòng/Book tour
        elif rand < 0.90: 
            dest = random.choice(destinations)
            book = random.choice(book_keywords)
            text = f"{book} ở {dest}" if random.random() > 0.5 else f"muốn {book} tại {dest}"
            
        # 10% dữ liệu: Hỏi tào lao / Lạc đề
        else: 
            text = random.choice(out_of_scope)
            
        # Nêm nếm thêm chút "teencode" ngẫu nhiên cho đời thêm mặn
        if random.random() > 0.8:
            text = text + random.choice([" wá", " z", " nha", " ko bot"])

        # Cấu trúc: (session_id, raw_text, predicted_intent, confidence, destination, budget, feedback)
        raw_data.append((f"sim_user_{i}", text, "unlabeled", 0.0, None, None, 0))

    return raw_data

def seed_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Dọn sạch DB cũ
        cur.execute("TRUNCATE TABLE ai_chat_analytics RESTART IDENTITY;")
        print("🧹 Đã dọn sạch bảng cũ.")

        # Sinh và bơm 1500 dòng mới
        data = generate_massive_data(1500)
        
        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget, user_feedback)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(sql, data)
        conn.commit()
        
        print(f"✅ Thành công! Đã bơm {len(data)} dòng dữ liệu khủng vào PostgreSQL.")

    except Exception as e:
        print(f"❌ Lỗi DB: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    seed_db()