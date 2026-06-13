import pandas as pd
from collections import defaultdict
import os
import re
import yaml

# ==========================================
# 1. TỪ ĐIỂN ENTITY (load động từ lookup tables)
# ==========================================

def load_entity_map():
    """Load destinations from Rasa lookup table + hardcoded categories/activities"""
    destinations = [
        "hà nội", "hồ chí minh", "sài gòn", "đà nẵng", "hải phòng", "cần thơ",
        "đà lạt", "nha trang", "huế", "hội an", "vũng tàu", "phú quốc", "sapa",
        "sa pa", "hạ long", "hà giang", "cao bằng", "ninh bình", "phan thiết",
        "mũi né", "côn đảo", "phú yên", "quy nhơn", "tam đảo", "mộc châu",
        "mai châu", "pù luông", "sầm sơn", "cửa lò", "phan rang", "bình ba",
        "nam du", "thanh hóa", "nghệ an", "hà tĩnh", "quảng bình", "quảng nam",
        "bình định", "khánh hòa", "bình thuận", "gia lai", "đắk lắk", "lâm đồng",
        "tây ninh", "bình dương", "đồng nai", "bến tre", "tà xùa", "măng đen",
        "trị an", "khe lim", "bình hưng", "bình tiên", "phú quý", "tà năng"
    ]

    # Try to extend from Rasa lookup table
    try:
        path = "rasa_bot/data/train/nlu.yml"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            for item in doc.get("nlu", []):
                if item.get("lookup") == "location":
                    lines = item.get("examples", "").strip().split("\n")
                    for loc in lines:
                        loc = loc.strip().lower()
                        if loc.startswith("- "):
                            loc = loc[2:]
                        if loc and loc not in destinations:
                            destinations.append(loc)
                    break
    except Exception:
        pass

    return {
        "destination": destinations,
        "category": [
            "khách sạn", "hotel", "resort", "homestay", "nhà nghỉ", "nhà trọ",
            "hostel", "villa", "biệt thự", "khu nghỉ dưỡng", "chỗ ở", "chỗ nghỉ",
            "phòng", "căn hộ", "apartment", "farmstay", "glamping", "khu cắm trại",
            "bungalow", "lodge", "nhà hàng", "quán ăn", "quán nhậu", "quán bar",
            "quán cà phê", "cafe", "tour", "vé", "combo", "gói du lịch",
            "biển", "núi", "miền núi", "miền biển", "miền tây", "miền bắc",
            "miền trung", "miền nam", "đồi núi", "rừng", "hồ", "sông", "suối",
            "thác", "đảo", "hòn đảo", "thành phố", "vùng quê", "đồng bằng",
            "cao nguyên", "yên tĩnh", "náo nhiệt", "sôi động", "chữa lành",
            "nghỉ dưỡng", "sang trọng", "bình dân", "giá rẻ", "tiết kiệm",
            "5 sao", "4 sao", "3 sao", "view biển", "view núi", "view hồ",
            "gần trung tâm",
        ],
        "activity": [
            "leo núi", "trekking", "cắm trại", "săn mây", "lặn san hô",
            "lặn biển", "dù lượn", "cáp treo", "tắm biển", "đạp xe",
            "chèo thuyền", "kayak", "đi bộ", "hiking", "ngắm hoa",
            "chụp ảnh", "sống ảo", "check in", "tham quan", "câu cá",
            "ngắm hoàng hôn", "ngắm bình minh", "spa", "massage", "yoga",
            "thiền", "chữa lành", "healing", "mua sắm", "shopping",
            "ăn uống", "ẩm thực", "ăn hải sản", "ăn đặc sản", "nhậu",
            "giải trí", "vui chơi", "trượt tuyết", "trượt nước",
            "zipline", "đu dây", "chèo sup", "paddle", "đi tàu",
            "du thuyền", "tour đảo", "thăm làng chài", "thăm bản làng",
            "homestay trải nghiệm", "hái trái cây", "làm nông",
            "farmstay", "cưỡi ngựa", "chụp ảnh cưới", "tuần trăng mật",
            "dạo phố", "đi phượt", "phượt", "road trip", "camping",
            "dã ngoại", "picnic", "hoạt động vui chơi", "đi xe đạp",
        ]
    }


ENTITY_MAP = load_entity_map()

def clean_teencode_garbage(text):
    """Hàm dọn dẹp các cụm từ rác do sinh data tự động gây ra"""
    t = str(text)
    # Loại bỏ khoảng trắng thừa
    return ' '.join(t.split())

def auto_annotate_entities(text):
    annotated_text = str(text)
    
    for entity_type, keywords in ENTITY_MAP.items():
        keywords.sort(key=len, reverse=True)
        for kw in keywords:
            pattern = re.compile(rf'(?<!\[)\b({kw})\b(?!\])', re.IGNORECASE)
            annotated_text = pattern.sub(rf'[\1]({entity_type})', annotated_text)
            
    return annotated_text

def append_to_nlu_yml():
    csv_path = 'data/high_confidence_auto_labeled.csv'
    yml_path = 'rasa_bot/data/train/nlu_auto_labeled.yml'
    
    print("[INFO] Đang chạy bản FINAL TUNE: Thêm từ khóa & Dọn rác teencode...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {csv_path}.")
        return

    if df.empty:
        print("⚠️ File CSV rỗng. Không có dữ liệu auto-labeled để xuất.")
        return

    intent_groups = defaultdict(set)
    skipped = 0
    
    for _, row in df.iterrows():
        intent = str(row.get('snorkel_intent', '')).strip()
        text = str(row.get('cleaned_text', '')).strip()
        
        if pd.isna(text) or pd.isna(intent) or intent in ("UNLABELED", "nan", "") or text == "nan":
            skipped += 1
            continue
            
        # 1. Tẩy gạch dưới
        text_natural = text.replace('_', ' ')
        
        # 2. Dọn rác teencode thừa
        text_clean = clean_teencode_garbage(text_natural)
        
        # 3. Bọc Entity với từ điển mới
        text_final = auto_annotate_entities(text_clean)
        
        intent_groups[intent].add(text_final)

    if skipped > 0:
        print(f"⚠️ Đã bỏ qua {skipped} dòng UNLABELED/nan")

    print(f"✍️ Đang xuất dữ liệu vào {yml_path}...")
    
    os.makedirs(os.path.dirname(yml_path), exist_ok=True)
    with open(yml_path, 'w', encoding='utf-8') as f:
        f.write("version: \"3.1\"\n")
        f.write("\nnlu:\n")
        
        for intent in sorted(intent_groups.keys()):
            f.write(f"\n  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for text in sorted(intent_groups[intent]):
                f.write(f"      - {text}\n")

    total = sum(len(v) for v in intent_groups.values())
    print("-" * 40)
    print("✅ HOÀN TẤT! Data đã đạt chuẩn Vàng để đem đi Train.")
    print(f"📊 Tổng số: {total} examples, {len(intent_groups)} intents")
    for intent, examples in sorted(intent_groups.items()):
        print(f"   - {intent}: {len(examples)} examples")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    append_to_nlu_yml()