# database/import_accommodations.py
import pandas as pd
import unicodedata
import random
from database.db_connection import get_connection

def determine_type(name: str) -> str:
    """Hàm tự động đoán loại hình lưu trú dựa vào tên"""
    name_lower = str(name).lower()
    
    if "resort" in name_lower or "khu nghỉ dưỡng" in name_lower or "khu nghi duong" in name_lower:
        return "resort"
    elif "homestay" in name_lower:
        return "homestay"
    elif "bungalow" in name_lower:
        return "bungalow"
    elif "villa" in name_lower:
        return "villa"
    elif "motel" in name_lower or "nha nghi" in name_lower or "nhà nghỉ" in name_lower:
        return "motel"
    elif "hotel" in name_lower or "khách sạn" in name_lower or "khach san" in name_lower:
        return "hotel"
    elif "hostel" in name_lower or "guesthouse" in name_lower:
        return "hostel"
    else:
        return "other"

def normalize_geo_name(geo_name: str) -> str:
    """Hàm chuẩn hóa tên địa danh: Xóa khoảng trắng, đưa về chuẩn Unicode Dựng Sẵn"""
    if pd.isna(geo_name):
        return ""
        
    # 1. Chuyển thành chuỗi và chuẩn hóa Unicode về dạng Dựng Sẵn (NFC)
    name_str = unicodedata.normalize('NFC', str(geo_name))
    
    # 2. Xóa khoảng trắng thừa ở 2 đầu và đưa về chữ thường để dễ so sánh
    name_lower = name_str.strip().lower()
    
    # 3. Gộp các biến thể
    if "vũng tàu" in name_lower or "vung tau" in name_lower or "bà rịa" in name_lower or "ba ria" in name_lower:
        return "Vũng Tàu"
        
    if "hồ chí minh" in name_lower or "ho chi minh" in name_lower or "sài gòn" in name_lower:
        return "Hồ Chí Minh"
        
    # Nếu không thuộc các nhóm trên, trả về tên gốc (đã được dọn dẹp)
    return name_str.strip()

def import_accommodations():
    print("[INFO] Đang kết nối Database để chèn Accommodations (Khách sạn)...")
    conn = get_connection()
    cur = conn.cursor()

    # 1. Đảm bảo bảng accommodations có cột 'type'
    try:
        cur.execute("ALTER TABLE accommodations ADD COLUMN IF NOT EXISTS type VARCHAR(50);")
        conn.commit()
        print("[INFO] Đã kiểm tra và thêm cột 'type' vào bảng accommodations.")
    except Exception as e:
        print(f"[ERROR] Lỗi khi thêm cột type: {e}")
        conn.rollback()

    # 2. Đọc file CSV
    csv_file = "data/hotel_coordinate.csv" 
    try:
        print(f"[INFO] Đang đọc dữ liệu từ file {csv_file}...")
        df_hotels = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file {csv_file}. Vui lòng kiểm tra lại đường dẫn.")
        cur.close()
        conn.close()
        return

    # 3. Bơm dữ liệu vào Database
    print(f"[INFO] Đang xử lý và chèn {len(df_hotels)} bản ghi...")
    success_count = 0

    for index, row in df_hotels.iterrows():
        hotel_id = row.get('locationId')
        name = row.get('name')
        lat = row.get('latitude')
        lng = row.get('longitude')
        geo_name_raw = row.get('parentGeo') # Lấy tên gốc từ CSV

        # 3.1. BẢO KÊ DỮ LIỆU: Chỉ cần có TÊN và ĐỊA DANH là nhận
        if pd.isna(name) or pd.isna(geo_name_raw):
            continue 

        # 3.2. CHUẨN HÓA TÊN ĐỊA DANH
        geo_name = normalize_geo_name(geo_name_raw)
        if not geo_name:
            continue

        # 3.3. AUTO-ID: Nếu mất locationId hoặc bị lỗi chữ, tự đẻ ra 1 số ngẫu nhiên
        if pd.isna(hotel_id):
            hotel_id = random.randint(900000000, 999999999)
        else:
            try:
                hotel_id = int(hotel_id) # Ép kiểu về số nguyên
            except ValueError:
                # Nếu cột ID bị lệch chứa chữ (vd: 'Khu nghỉ dưỡng'), thì cấp ID ảo luôn
                hotel_id = random.randint(900000000, 999999999)

        # Xác định type
        acc_type = determine_type(name)

        # 3.4. Tìm hoặc Tự Động Đẻ Địa Danh
        cur.execute("SELECT id FROM destinations WHERE location ILIKE %s LIMIT 1", (f"%{geo_name}%",))
        result = cur.fetchone()

        if result:
            destination_id = result[0]
        else:
            print(f"[WARN] Địa danh '{geo_name}' chưa có. Đang tự động tạo...")
            cur.execute("""
                INSERT INTO destinations (location, description, price_level) 
                VALUES (%s, %s, %s) RETURNING id;
            """, (
                geo_name, 
                "Dữ liệu tự động sinh từ danh sách Khách sạn.", 
                "Chưa xác định"
            ))
            destination_id = cur.fetchone()[0]

        # 3.5. Insert hoặc Update vào bảng accommodations
        lat_val = None if pd.isna(lat) else lat
        lng_val = None if pd.isna(lng) else lng

        cur.execute("""
            INSERT INTO accommodations (id, destination_id, name, type, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                type = EXCLUDED.type,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude;
        """, (hotel_id, destination_id, name, acc_type, lat_val, lng_val))
        success_count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"[INFO] HOÀN TẤT! Đã chèn/cập nhật {success_count} bản ghi.")

if __name__ == "__main__":
    import_accommodations()