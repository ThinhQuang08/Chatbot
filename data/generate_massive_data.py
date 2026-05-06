import psycopg2
import random

DB_CONFIG = {
    "host": "localhost",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def generate_fine_grained_data(num_samples=2000):
    destinations = ["đà lạt", "sapa", "nha trang", "phú quốc", "đà nẵng", "hà nội", "sài gòn", "đlạt", "nt", "pq"]
    
    # Các mẫu câu đặc thù cho từng Intent
    patterns = {
        "search_destination": [
            "review {dest} mùa này", "xin kinh nghiệm đi {dest}", "thời tiết {dest} sao", 
            "{dest} có chỗ nào check in đẹp", "đặc sản {dest} là gì", "mùa nào đi {dest} hợp lý"
        ],
        "search_activity": [
            "ở {dest} có tour lặn không", "muốn đi trekking {dest}", "có cắm trại ở {dest} k",
            "chơi dù lượn ở {dest}", "các hoạt động vui chơi tại {dest}", "thuê xe đạp dạo {dest}"
        ],
        "search_price": [
            "giá tour {dest} nhiêu tiền", "chi phí đi {dest} mấy củ", "tour {dest} rẻ nhất là bao nhiêu",
            "có khuyến mãi tour {dest} không", "xin bảng giá đi {dest}"
        ],
        "ask_tour_info": [
            "xin lịch trình đi {dest}", "tour {dest} bao gồm ăn uống chưa", "di chuyển bằng phương tiện gì",
            "tour {dest} mấy ngày mấy đêm", "có hướng dẫn viên không", "gửi mình chi tiết tour {dest}"
        ],
        "search_accommodation": [
            "đặt phòng khách sạn {dest}", "thuê resort {dest}", "tìm homestay view đẹp ở {dest}",
            "book phòng {dest}", "tìm chỗ ở giá rẻ tại {dest}"
        ],
        "search_travel": [
            "tìm tour đi {dest}", "muốn đi {dest}", "có tour {dest} không bot", 
            "tư vấn tour {dest}", "ik {dest} chơi"
        ],
        "out_of_scope": [
            "cách nấu bún bò", "bot ngu vãi", "thời tiết hôm nay", "bạn tên gì", 
            "buồn ngủ quá", "giải dùm bài toán", "chửi thề dmm"
        ]
    }

    raw_data = []
    
    print(f"[INFO] Đang trộn từ vựng để sinh ra {num_samples} câu chat đa dạng...")
    for i in range(num_samples):
        # Chọn ngẫu nhiên một Intent category (tỷ lệ random)
        rand = random.random()
        if rand < 0.15: category = "search_destination"
        elif rand < 0.30: category = "search_activity"
        elif rand < 0.50: category = "search_price"
        elif rand < 0.65: category = "ask_tour_info"
        elif rand < 0.80: category = "search_accommodation"
        elif rand < 0.95: category = "search_travel"
        else: category = "out_of_scope"

        # Lấy template và nhét tên địa danh vào
        template = random.choice(patterns[category])
        dest = random.choice(destinations)
        text = template.replace("{dest}", dest)
            
        # Thêm gia vị Teencode / Sai chính tả (20% xác suất)
        if random.random() > 0.8:
            text = text.replace("đi", "ik").replace("không", "ko").replace("quá", "wá")
            text += random.choice([" nha", " z", " dợ", " bot"])

        # Cấu trúc: (session_id, raw_text, predicted_intent, confidence, destination, budget, feedback)
        raw_data.append((f"sim_v2_{i}", text, "unlabeled", 0.0, None, None, 0))

    return raw_data

def seed_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE ai_chat_analytics RESTART IDENTITY;")
        print("🧹 Đã dọn sạch bảng cũ.")

        data = generate_fine_grained_data(2000) # Tăng lên 2000 câu cho mô hình học sướng
        
        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget, user_feedback)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(sql, data)
        conn.commit()
        
        print(f"✅ Bơm thành công {len(data)} dòng dữ liệu siêu đa dạng vào PostgreSQL!")

    except Exception as e:
        print(f"❌ Lỗi DB: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    seed_db()