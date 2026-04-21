import json
from database.db_connection import get_connection

def import_tours():
    print("[INFO] Đang kết nối Database để chèn Tours...")
    conn = get_connection()
    cur = conn.cursor()

    json_file = "data/data_normalized_tour.json"
    with open(json_file, "r", encoding="utf-8") as f:
        tours = json.load(f)

    print(f"[INFO] Đang chèn {len(tours)} tour và liên kết địa danh...")
    for t in tours:
        entities = t.get("rasa_entities", {})
        departure_list = entities.get("departure", [])
        departure = departure_list[0] if departure_list else ""
        tour_id = t["tour_id"]
        
        # 1. Insert vào bảng tours (Đã bỏ cột destination JSON)
        cur.execute("""
            INSERT INTO tours (
                tour_id, tour_name, departure, price, currency,
                days, nights, duration_text, transportation, rag_knowledge_base,
                season_tag, data_source, is_vectorized
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tour_id) DO UPDATE SET
                tour_name = EXCLUDED.tour_name,
                price = EXCLUDED.price,
                rag_knowledge_base = EXCLUDED.rag_knowledge_base;
        """, (
            tour_id,
            t["tour_name"],
            departure,
            entities.get("price", {}).get("normalized_value", 0),
            entities.get("price", {}).get("currency", "VND"),
            entities.get("duration", {}).get("days", 0),
            entities.get("duration", {}).get("nights", 0),
            entities.get("duration", {}).get("raw_text", ""),
            entities.get("transportation", ""),
            json.dumps(t.get("rag_knowledge_base", {}), ensure_ascii=False),
            t.get("mlops_metadata", {}).get("season_tag", "All_Season"),
            t.get("mlops_metadata", {}).get("data_source", "Vietravel"),
            t.get("mlops_metadata", {}).get("is_vectorized", False)
        ))

        # 2. Xử lý logic Nối Bảng (Bảng trung gian tour_destinations)
        dest_list = entities.get("destination", [])
        for d_name in dest_list:
            if not d_name: continue # Bỏ qua nếu tên rỗng
            
            # Tìm ID của địa danh (Dùng dấu % để tìm tương đối)
            cur.execute("SELECT id FROM destinations WHERE location ILIKE %s LIMIT 1", (f"%{d_name}%",))
            res = cur.fetchone()
            
            if res:
                dest_id = res[0]
            else:
                # ==========================================
                # 🔥 AUTO-CREATE: Tự động đẻ ra địa danh mới nếu chưa có!
                # ==========================================
                print(f"[WARN] Địa danh '{d_name}' chưa có trong DB. Hệ thống đang tự động tạo...")
                
                # Cần điền các trường NOT NULL (description, price_level) bằng giá trị mặc định
                cur.execute("""
                    INSERT INTO destinations (location, description, price_level) 
                    VALUES (%s, %s, %s) RETURNING id;
                """, (
                    d_name, 
                    "Dữ liệu tự động sinh từ hệ thống Crawler. Đang chờ cập nhật chi tiết.", 
                    "Chưa xác định"
                ))
                dest_id = cur.fetchone()[0]

            # Nối Tour và Destination vào bảng trung gian
            cur.execute("""
                INSERT INTO tour_destinations (tour_id, destination_id) 
                VALUES (%s, %s) ON CONFLICT DO NOTHING;
            """, (tour_id, dest_id))

    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] HOÀN TẤT import Tours và ánh xạ Tour-Destination!")

if __name__ == "__main__":
    import_tours()