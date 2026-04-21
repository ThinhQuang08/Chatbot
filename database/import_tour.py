# database/import_tour.py
import json
from database.db_connection import get_connection

def import_tours():
    print("Đang kết nối Database...")
    conn = get_connection()
    cur = conn.cursor()

    # 1. Tạo bảng tours (nếu chưa có)
    print("Đang khởi tạo cấu trúc bảng 'tours'...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tours (
            tour_id VARCHAR(50) PRIMARY KEY,
            tour_name TEXT,
            departure VARCHAR(100),
            destination JSONB,
            price INTEGER,
            currency VARCHAR(10),
            days INTEGER,
            nights INTEGER,
            duration_text VARCHAR(100),
            transportation VARCHAR(100),
            rag_knowledge_base JSONB,
            season_tag VARCHAR(50),
            data_source VARCHAR(50),
            is_vectorized BOOLEAN DEFAULT FALSE
        )
    """)

    # 2. Đọc file JSON chuẩn hóa
    json_file = "data/data_normalized_tour.json"
    with open(json_file, "r", encoding="utf-8") as f:
        tours = json.load(f)

    # 3. Insert dữ liệu vào Database
    print(f"Đang chèn {len(tours)} tour vào Database...")
    for t in tours:
        entities = t.get("rasa_entities", {})
        departure_list = entities.get("departure", [])
        departure = departure_list[0] if departure_list else ""
        
        # Dùng ON CONFLICT để nếu chạy lại file này nhiều lần sẽ không bị lỗi trùng ID (Nó sẽ Update)
        cur.execute("""
            INSERT INTO tours (
                tour_id, tour_name, departure, destination, price, currency,
                days, nights, duration_text, transportation, rag_knowledge_base,
                season_tag, data_source, is_vectorized
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tour_id) DO UPDATE SET
                tour_name = EXCLUDED.tour_name,
                price = EXCLUDED.price,
                rag_knowledge_base = EXCLUDED.rag_knowledge_base,
                is_vectorized = FALSE;
        """, (
            t["tour_id"],
            t["tour_name"],
            departure,
            json.dumps(entities.get("destination", []), ensure_ascii=False), # Chuyển List thành JSONB
            entities.get("price", {}).get("normalized_value", 0),
            entities.get("price", {}).get("currency", "VND"),
            entities.get("duration", {}).get("days", 0),
            entities.get("duration", {}).get("nights", 0),
            entities.get("duration", {}).get("raw_text", ""),
            entities.get("transportation", ""),
            json.dumps(t.get("rag_knowledge_base", {}), ensure_ascii=False), # Chuyển Dict thành JSONB
            t.get("mlops_metadata", {}).get("season_tag", "All_Season"),
            t.get("mlops_metadata", {}).get("data_source", "Vietravel"),
            t.get("mlops_metadata", {}).get("is_vectorized", False)
        ))
    
    conn.commit()
    cur.close()
    conn.close()
    print("HOÀN TẤT! Dữ liệu đã được import trong PostgreSQL.")

if __name__ == "__main__":
    import_tours()