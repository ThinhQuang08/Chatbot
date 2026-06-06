import yaml
import re
import pandas as pd
import os

# Giả sử đây là hàm tách từ (Word Segmentation) mà sếp đang dùng trong file preprocess_data.py
# Sếp hãy import hàm thật của sếp vào đây nhé (ví dụ: from underthesea import word_tokenize)
def mock_underthesea_segment(text):
    # Đây chỉ là hàm mô phỏng. Sếp SẼ THAY BẰNG HÀM THẬT CỦA SẾP.
    # Mục đích là để "Nha Trang" biến thành "Nha_Trang" y như file production.
    # return word_tokenize(text, format="text") 
    return text 

def clean_rasa_entity(text):
    """
    Hàm này lột vỏ entity của Rasa.
    Ví dụ: 'chi phí đi [Phú Quốc](destination) mấy triệu' -> 'chi phí đi Phú Quốc mấy triệu'
    """
    # Dùng Regex tìm tất cả các chuỗi dạng [text](label) và chỉ lấy phần 'text'
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return cleaned.strip()

def create_reference_csv(yml_path="rasa_bot/data/train/nlu.yml", output_csv="data/reference_data.csv"):
    print(f"📖 Đang đọc dữ liệu từ {yml_path}...")
    
    with open(yml_path, 'r', encoding='utf-8') as f:
        nlu_data = yaml.safe_load(f)

    records = []
    
    # Lặp qua từng khối intent trong file YAML
    for item in nlu_data.get('nlu', []):
        if 'intent' in item:
            intent_name = item['intent']
            examples = item.get('examples', '')
            
            if examples:
                # Cắt từng dòng example
                for line in examples.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        # Lấy nội dung câu chat, bỏ dấu "- " ở đầu
                        raw_text = line[2:]
                        
                        # 1. Bỏ vỏ bọc Entity của Rasa
                        text_no_entity = clean_rasa_entity(raw_text)
                        
                        # 2. Chạy qua hàm chuẩn hóa/tách từ (ĐỂ KHỚP VỚI PRODUCTION)
                        final_cleaned_text = mock_underthesea_segment(text_no_entity)
                        
                        records.append({
                            'cleaned_text': final_cleaned_text,
                            'predicted_intent': intent_name,
                            'confidence_score': 1.0 # Dữ liệu chuẩn luôn tự tin 100%
                        })

    # Xuất ra file CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Tuyệt vời! Đã xuất {len(df)} câu chuẩn ra file: {output_csv}")
    print(df.head())

if __name__ == "__main__":
    create_reference_csv()