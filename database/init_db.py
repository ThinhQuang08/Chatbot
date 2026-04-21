from database.db_connection import get_connection

def init_database():
    print("[INFO] Đang khởi tạo cấu trúc Database chuẩn hóa (PK/FK)...")
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Tạo bảng Destinations (Bảng cha)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS destinations (
                id SERIAL PRIMARY KEY,
                location TEXT NOT NULL UNIQUE, -- Thêm UNIQUE để dễ truy vấn map ID
                season TEXT,
                description TEXT NOT NULL,
                activities TEXT,
                avg_cost TEXT,
                price_level TEXT NOT NULL,
                season_start INTEGER,
                season_end INTEGER,
                avg_cost_min INTEGER,
                avg_cost_max INTEGER,
                activities_list TEXT[] NOT NULL DEFAULT '{}'
            );
        """)

        # Tạo bảng Tours
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tours (
                tour_id VARCHAR(50) PRIMARY KEY,
                tour_name TEXT,
                departure VARCHAR(100),
                price INTEGER,
                currency VARCHAR(10) DEFAULT 'VND',
                days INTEGER,
                nights INTEGER,
                duration_text VARCHAR(100),
                transportation VARCHAR(100),
                rag_knowledge_base JSONB,
                season_tag VARCHAR(50),
                data_source VARCHAR(50),
                is_vectorized BOOLEAN DEFAULT FALSE
            );
        """)

        # Tạo bảng trung gian Tour_Destinations (N-N)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tour_destinations (
                tour_id VARCHAR(50),
                destination_id INTEGER,
                PRIMARY KEY (tour_id, destination_id),
                FOREIGN KEY (tour_id) REFERENCES tours(tour_id) ON DELETE CASCADE,
                FOREIGN KEY (destination_id) REFERENCES destinations(id) ON DELETE CASCADE
            );
        """)

        # Tạo bảng Accommodations (Khách sạn)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS accommodations (
                id BIGINT PRIMARY KEY,
                destination_id INTEGER,
                name VARCHAR(255) NOT NULL,
                latitude FLOAT,
                longitude FLOAT,
                FOREIGN KEY (destination_id) REFERENCES destinations(id) ON DELETE CASCADE
            );
        """)

        conn.commit()
        print("[INFO] Đã tạo xong toàn bộ Schema!")
    except Exception as e:
        print(f"[ERROR] Lỗi tạo bảng: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    init_database()