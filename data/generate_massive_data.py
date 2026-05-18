import psycopg2
import random

DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def generate_tourism_drift(num_samples=1500):
    # 1. Tập trung vào các ĐỊA ĐIỂM MỚI NỔI (Không hề có trong nlu.yml)
    new_destinations = ["Măng Đen", "Tà Xùa", "Phú Quý", "Hà Giang", "Trị An", "Tà Năng"]
    
    # 2. Tập trung vào XU HƯỚNG MỚI (Glamping, Chữa lành, Trekking)
    trend_patterns = [
        "cho mình xin báo giá glamping ở {dest} cho {people} người",
        "có tour chữa lành nào đi {dest} khoảng {days} ngày không",
        "mình muốn tìm bãi cắm trại staycation gần {dest} đi {days} ngày",
        "review đi săn mây ở {dest} {days} ngày {days_minus} đêm",
        "chi phí đi trekking băng rừng ở {dest} cho nhóm {people} người",
        "tư vấn lịch trình đi {dest} tự túc {days} ngày",
        "ở {dest} có khu glamping nào view đẹp không",
        "mình cần tìm nơi yên tĩnh để nghỉ dưỡng chữa lành tại {dest}"
    ]

    # 3. Một số câu hỏi chuyên sâu về nghiệp vụ mà bot chưa rành
    deep_tour_patterns = [
        "tour {dest} có bao gồm vé xe giường nằm cabin đôi không",
        "resort ở {dest} có cho mang theo chó mèo thú cưng không",
        "nhóm mình {people} người lớn {kids} trẻ em đi {dest} thì phụ thu sao",
        "người lớn tuổi ngồi xe lăn có đi tour {dest} {days} ngày được không"
    ]

    raw_data = []
    
    print(f"[INFO] Đang giả lập làn sóng {num_samples} khách hàng đu trend du lịch mới...")
    
    for i in range(num_samples):
        people = random.randint(2, 15)
        kids = random.randint(1, 4)
        days = random.randint(2, 5)
        
        # Chọn ngẫu nhiên giữa xu hướng mới và câu hỏi nghiệp vụ khó
        if random.random() < 0.7:
            template = random.choice(trend_patterns)
        else:
            template = random.choice(deep_tour_patterns)
            
        dest = random.choice(new_destinations)
        
        # Điền thông số để tạo câu unique (qua mặt MD5)
        text = template.format(dest=dest, people=people, days=days, days_minus=days-1, kids=kids)

        # --- MÔ PHỎNG SỰ BỐI RỐI CỦA MÔ HÌNH ---
        # Bot thấy từ khóa lạ (glamping, chữa lành, Tà Xùa) nên tự tin giảm mạnh
        predicted_intent = random.choice(["search_activity", "search_travel", "search_price", "ask_tour_info"])
        
        # Độ tự tin rất "nửa vời" (0.45 - 0.75) vì câu hỏi có cấu trúc du lịch nhưng từ vựng lạ
        confidence = round(random.uniform(0.45, 0.75), 2) 

        # Cấu trúc: (session_id, raw_text, predicted_intent, confidence, destination, budget, feedback)
        raw_data.append((f"trend_user_{i}", text, predicted_intent, confidence, None, None, 0))

    return raw_data

def seed_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE ai_chat_analytics RESTART IDENTITY;")
        print("🧹 Đã dọn sạch bảng cũ.")

        data = generate_tourism_drift(1500) 
        
        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget, user_feedback)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(sql, data)
        conn.commit()
        
        print(f"✅ Bơm thành công {len(data)} câu hỏi xu hướng chuyên môn vào DB!")

    except Exception as e:
        print(f"❌ Lỗi DB: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    seed_db()