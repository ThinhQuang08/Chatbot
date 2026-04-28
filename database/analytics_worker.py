import pika
import json
import psycopg2
import re
# 1. Cấu hình kết nối
RABBITMQ_HOST = 'localhost'
QUEUE_NAME = 'rasa_analytics_queue'

DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

def parse_budget_vnd(text_value):
    if not text_value:
        return None
        
    text_value = str(text_value).lower().strip()
    
    # Xóa dấu chấm, phẩy, khoảng trắng (VD: "5.000.000" -> "5000000")
    clean_text = re.sub(r'[,.\s]', '', text_value)
    
    try:
        if 'triệu' in text_value or 'tr' in text_value or 'củ' in text_value:
            number_part = re.sub(r'[^\d]', '', clean_text)
            return int(number_part) * 1000000 if number_part else None
            
        elif 'k' in text_value or 'ngàn' in text_value or 'nghìn' in text_value:
            number_part = re.sub(r'[^\d]', '', clean_text)
            return int(number_part) * 1000 if number_part else None
            
        else:
            number_part = re.sub(r'[^\d]', '', clean_text)
            return int(number_part) if number_part else None
    except Exception as e:
        print(f"[WARN] Không thể parse ngân sách: {text_value}")
        return None

def insert_to_analytics_db(event_data):
    """Hàm ghi dữ liệu sạch vào Database Phân tích"""
    # Chỉ quan tâm đến sự kiện user gửi tin nhắn (Bỏ qua các log hệ thống lằng nhằng)
    if event_data.get("event") != "user":
        return

    # Trích xuất dữ liệu từ cục JSON khổng lồ của Rasa
    session_id = event_data.get("sender_id")
    raw_text = event_data.get("text")
    
    parse_data = event_data.get("parse_data", {})
    intent = parse_data.get("intent", {}).get("name")
    confidence = parse_data.get("intent", {}).get("confidence")
    
    # Lấy các Entities (Slot) để phục vụ Evidently AI
    entities = parse_data.get("entities", [])
    destination = next((e['value'] for e in entities if e['entity'] == 'destination'), None)

    raw_budget = next((e['value'] for e in entities if e['entity'] == 'budget'), None)
    parsed_budget_value = parse_budget_vnd(raw_budget)

    print(f"[WORKER] Nhận tin nhắn '{raw_text}' từ {session_id}: Intent={intent} ({confidence}) | Budget: {parsed_budget_value}")

    # Lưu vào PostgreSQL
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Insert vào cái bảng chúng ta thiết kế ở Bước trước
        sql = """
            INSERT INTO ai_chat_analytics 
            (session_id, raw_text, predicted_intent, confidence_score, destination, parsed_budget)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(sql, (session_id, raw_text, intent, confidence, destination, parsed_budget_value))
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR DB] {e}")

def callback(ch, method, properties, body):
    """Hàm được gọi mỗi khi RabbitMQ có tin nhắn mới"""
    try:
        event_data = json.loads(body)
        insert_to_analytics_db(event_data)
    except Exception as e:
        print(f"[ERROR PARSE] {e}")

def start_worker():
    """Khởi động Worker lắng nghe 24/7"""
    print("[INFO] Đang kết nối tới RabbitMQ...")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()

    # Đảm bảo queue tồn tại
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    print(f"[*] Worker đang lắng nghe log tại queue '{QUEUE_NAME}'. Bấm CTRL+C để thoát.")
    
    # Bắt đầu nghe
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

if __name__ == "__main__":
    start_worker()