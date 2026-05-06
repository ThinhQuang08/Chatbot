import pandas as pd
from collections import defaultdict
import os
import re

# ==========================================
# 1. TỪ ĐIỂN CẬP NHẬT: THÊM CÁC CỤM TỪ BỊ LỌT
# ==========================================
ENTITY_MAP = {
    "destination": ["đà lạt", "nha trang", "phú quốc", "sapa", "sa pa", "đà nẵng", "hà nội", "sài gòn", "vũng tàu", "hội an"],
    # Đã thêm "chỗ ở" vào category
    "category": ["khách sạn", "resort", "homestay", "nhà nghỉ", "nhà hàng", "nghỉ dưỡng", "villa", "tour", "chỗ ở"],
    # Đã thêm cụm "hoạt động vui chơi" vào activity
    "activity": ["cắm trại", "trekking", "leo núi", "đạp xe", "xe đạp", "dù lượn", "lặn", "check in", "tour đảo", "hoạt động vui chơi"]
}

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
    yml_path = 'data/nlu_test.yml'
    
    print("[INFO] Đang chạy bản FINAL TUNE: Thêm từ khóa & Dọn rác teencode...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {csv_path}.")
        return

    intent_groups = defaultdict(set)
    
    for _, row in df.iterrows():
        intent = str(row['snorkel_intent']).strip()
        text = str(row['cleaned_text']).strip()
        
        if pd.isna(text) or pd.isna(intent) or intent == "UNLABELED" or text == "nan":
            continue
            
        # 1. Tẩy gạch dưới
        text_natural = text.replace('_', ' ')
        
        # 2. Dọn rác teencode thừa
        text_clean = clean_teencode_garbage(text_natural)
        
        # 3. Bọc Entity với từ điển mới
        text_final = auto_annotate_entities(text_clean)
        
        intent_groups[intent].add(text_final)

    print(f"✍️ Đang xuất dữ liệu hoàn hảo vào {yml_path}...")
    
    with open(yml_path, 'a', encoding='utf-8') as f:
        f.write("\n\n# --- DỮ LIỆU ĐÃ AUTO-LABEL VÀ BỌC ENTITY (FINAL TUNE) ---\n")
        
        for intent, examples in intent_groups.items():
            f.write(f"  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for text in examples:
                f.write(f"      - {text}\n")

    print("-" * 40)
    print("✅ HOÀN TẤT! Data đã đạt chuẩn Vàng để đem đi Train.")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    append_to_nlu_yml()