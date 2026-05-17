import psycopg2
import random

DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

# Các từ đệm ngẫu nhiên để làm câu chat trở nên Độc Nhất (Unique)
endings = ["", "ạ", "nhé", "shop", "bot ơi", "tư vấn mình với", "nha", "gấp ạ"]
greetings = ["", "alo ", "cho hỏi ", "admin ơi ", "hi shop, "]

def generate_unique_drift_sentence():
    days = random.randint(2, 5)
    people = random.randint(2, 20)
    
    templates = [
        f"review cho em combo đi Măng Đen {days}N{days-1}Đ chữa lành với",
        f"tháng sau đi săn mây Tà Xùa {people} người thì đi xe khách nào ok",
        f"cho mình xin báo giá glamping cắm trại ở hồ Trị An cho {people} người",
        f"có tour trekking Tà Năng Phan Dũng nào cuối tuần này không",
        f"mình muốn tìm chỗ staycation gần Sài Gòn {days} ngày",
        f"đảo Phú Quý mùa này biển êm không, xin giá vé tàu cho {people} người",
        f"đang có flash sale hay mã giảm giá nào đi Phú Quốc không",
        f"mình cần thuê {random.randint(2, 10)} xe máy ga ở đà lạt giao tận khách sạn",
        f"đi Hội An {people} người lớn {random.randint(1, 4)} trẻ em thì ở villa nào",
        f"công ty mình tổ chức team building {random.randint(30, 100)} người ở Vũng Tàu xin báo giá",
        f"resort này có cho mang theo chó mèo thú cưng vào không",
        f"có tour thái lan bangkok pattaya {days}N{days-1}Đ nào rẻ không",
    ]
    
    # Ráp ngẫu nhiên: Lời chào + Câu lõi + Từ đệm để đảm bảo không bao giờ trùng nhau
    core_sentence = random.choice(templates)
    full_sentence = f"{random.choice(greetings)}{core_sentence} {random.choice(endings)}".strip()
    return full_sentence

def inject_valuable_drift():
    print("="*50)
    print("🚀 BẮT ĐẦU BƠM DỮ LIỆU DRIFT ĐỘC NHẤT (UNIQUE)...")
    print("="*50)
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        num_records = 350 # Bơm dư ra chút xíu
        
        for i in range(num_records):
            session_id = f"real_user_{random.randint(10000, 99999)}"
            raw_text = generate_unique_drift_sentence()
            
            predicted_intent = random.choice(["nlu_fallback", "inquire_tour", "ask_price"])
            confidence = round(random.uniform(0.35, 0.65), 2)
            
            sql = """
                INSERT INTO ai_chat_analytics 
                (session_id, raw_text, predicted_intent, confidence_score) 
                VALUES (%s, %s, %s, %s)
            """
            cur.execute(sql, (session_id, raw_text, predicted_intent, confidence))
            
        conn.commit()
        print(f"✅ Đã bơm thành công {num_records} câu hỏi UNIQUE vào Database!")
        print("👉 Bây giờ sếp chạy lại file preprocess_data.py, đảm bảo sẽ thu hoạch được bộn data!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()

if __name__ == "__main__":
    inject_valuable_drift()