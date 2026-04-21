import json
import re

INPUT_FILE = "data/tour.json"
OUTPUT_FILE = "data/data_normalized_tour.json"

def parse_price(price_str):
    """Chuyển đổi chuỗi giá (VD: '750,000') thành số nguyên."""
    if not price_str:
        return 0
    clean_str = re.sub(r'[^\d]', '', price_str)
    return int(clean_str) if clean_str else 0

def parse_duration(duration_str):
    """Bóc tách số ngày và đêm từ chuỗi (VD: '4N3Đ' hoặc 'Trong ngày')."""
    duration_str = str(duration_str).strip().lower()
    
    if "trong ngày" in duration_str:
        return 1, 0
    
    # Tìm mẫu như 4N3Đ, 4 Ngày 3 Đêm, 4n3d
    match = re.search(r'(\d+)\s*[nNngNgàyay]+\s*(\d+)\s*[đĐdĐêmêm]+', duration_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Nếu chỉ ghi số ngày (VD: 3 ngày)
    match_day_only = re.search(r'(\d+)\s*[nNngNgàyay]+', duration_str)
    if match_day_only:
        return int(match_day_only.group(1)), 0
        
    return 0, 0

def extract_destinations(tour_name, departure):
    """Trích xuất điểm đến từ tên tour (Dựa theo format của Vietravel)."""
    # Vietravel thường đặt tên tour theo dạng: Điểm xuất phát - Điểm đến 1 - Điểm đến 2
    parts = [p.strip() for p in tour_name.split("-")]
    
    # Bỏ phần tử đầu tiên nếu nó trùng với nơi khởi hành
    destinations = []
    for part in parts:
        if part.lower() != departure.lower() and part not in destinations:
            destinations.append(part)
            
    return destinations

def extract_itinerary(chi_tiet_lich_trinh):
    """Lọc ra các ngày lịch trình và đưa vào mảng."""
    itinerary = []
    for key, value in chi_tiet_lich_trinh.items():
        if key.lower().startswith("ngày"):
            itinerary.append({
                "title": key.strip(),
                "description": value.strip()
            })
    return itinerary

def convert_tour_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_data = []

    for item in data:
        chi_tiet = item.get("Chi_Tiet_Lich_Trinh", {})
        
        # Tiền xử lý các entity
        price_val = parse_price(item.get("Gia", ""))
        days, nights = parse_duration(item.get("Thoi_Gian", ""))
        departure = item.get("Khoi_Hanh", "")
        destinations = extract_destinations(item.get("Ten_Tour", ""), departure)

        # Cấu trúc lại thành format chuẩn cho RAG/MongoDB
        new_item = {
            "tour_id": item.get("Ma_Tour"),
            "tour_name": item.get("Ten_Tour"),
            "rasa_entities": {
                "departure": [departure] if departure else [],
                "destination": destinations,
                "category": [],
                "price": {
                    "raw_text": item.get("Gia"),
                    "normalized_value": price_val,
                    "currency": "VND"
                },
                "duration": {
                    "raw_text": item.get("Thoi_Gian"),
                    "days": days,
                    "nights": nights
                },
                "transportation": item.get("Phuong_Tien", "")
            },
            "rag_knowledge_base": {
                "itinerary": extract_itinerary(chi_tiet),
                "inclusions": chi_tiet.get("Giá tour bao gồm", "").split("\n"),
                "exclusions": chi_tiet.get("Giá tour không bao gồm", "").split("\n"),
                "children_policy": chi_tiet.get("Lưu ý giá trẻ em", "Tạm thời không có thông tin"),
                "payment_policy": chi_tiet.get("Điều kiện thanh toán", ""),
                "registration_policy": chi_tiet.get("Điều kiện đăng ký", ""),
                "cancellation_policy": chi_tiet.get("Các điều kiện hủy tour đối với ngày thường", "") 
                                       + "\n" + chi_tiet.get("Lưu ý về chuyển hoặc hủy tour", ""),
                "force_majeure": chi_tiet.get("Trường hợp bất khả kháng", ""),
                "contact_info": chi_tiet.get("Liên hệ", "")
            },
            "original_link": item.get("Link_Goc", ""),
            "mlops_metadata": {
                "season_tag": "All_Season",
                "data_source": "Crawled_Vietravel",
                "is_vectorized": False
            }
        }

        # Dọn dẹp các mảng rỗng do split("\n") sinh ra nếu data gốc rỗng
        new_item["rag_knowledge_base"]["inclusions"] = [i.strip() for i in new_item["rag_knowledge_base"]["inclusions"] if i.strip()]
        new_item["rag_knowledge_base"]["exclusions"] = [i.strip() for i in new_item["rag_knowledge_base"]["exclusions"] if i.strip()]

        normalized_data.append(new_item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)

    print(f"Đã chuẩn hóa thành công {len(normalized_data)} tours!")
    print(f"File lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_tour_data()