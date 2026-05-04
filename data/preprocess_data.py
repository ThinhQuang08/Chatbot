# data/preprocess_data.py
# A. Làm sạch và chuẩn hoá, tách từ

import pandas as pd
import psycopg2
import unicodedata
import re
import hashlib
import warnings
from underthesea import word_tokenize
warnings.filterwarnings('ignore')

DB_CONFIG = {
    "host": "192.168.1.213",
    "database": "chatbot",
    "user": "chatbot_user",
    "password": "supersecret"
}

class DataPreprocessor:
    def __init__(self):
        # 1. TỪ ĐIỂN CỤM TỪ (Phrase Mapping) - Chạy trước tiên để bắt các cụm dài & Không dấu
        self.phrase_dict = {
            "pù cúp": "Phú Quốc",
            "da nang": "Đà Nẵng",
            "nha trang": "Nha Trang",
            "sa pa": "Sa Pa",
            "sapa": "Sa Pa",
            "da lat": "Đà Lạt",
            "đà lạt": "Đà Lạt",
            "phu quoc": "Phú Quốc",
            "phuu quoc": "Phú Quốc",
            "gia re": "giá rẻ",
            "khach san": "khách sạn"
        }
        
        # 2. TỪ ĐIỂN TỪ ĐƠN (Word Mapping) - Chạy sau khi đã tách cụm
        self.word_dict = {
            "củ": "triệu", "tr": "triệu", "k": "nghìn",
            "đl": "Đà Lạt", "đlạt": "Đà Lạt", "nt": "Nha Trang", "pq": "Phú Quốc",
            "hn": "Hà Nội", "sg": "Sài Gòn",
            "ik": "đi", "dc": "được", "đc": "được", "ko": "không", "k": "không",
            "vjp": "vip", "resot": "resort", "xĩu": "xỉu",
            
            # --- CÁC TỪ MỚI BỔ SUNG ĐỂ QUÉT SẠCH 30 CÂU TEST ---
            "mún": "muốn", "chjll": "chill", "way": "quay", "ks": "khách sạn",
            "thág": "tháng", "z": "vậy", "nhiu": "nhiêu", "tien": "tiền",
            "khach": "khách", "sann": "sạn", "dep": "đẹp", "wá": "quá",
            "tim": "tìm", "di": "đi", "gia": "giá", "re": "rẻ"
        }

    def generate_md5(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def normalize_unicode(self, text):
        return unicodedata.normalize('NFC', text)

    def clean_text(self, text):
        text = text.strip()
        
        # Xử lý chữ "k" đi liền với số TRƯỚC (VD: 200k, 50k)
        text = re.sub(r'(\d+)\s*k\b', r'\1 nghìn', text, flags=re.IGNORECASE)
        
        # Tách số và chữ cho các trường hợp còn lại (VD: "2tr" -> "2 tr")
        text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)
        
        # Xóa ký tự lặp (VD: đẹpppp -> đẹp)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        return text

    def phrase_normalize(self, text):
        """VŨ KHÍ 2: Quét và thay thế nguyên cụm từ trước khi cắt nhỏ"""
        # Quét qua các cụm từ trong từ điển (Dùng Regex thay thế để không bị lỗi chữ hoa/thường)
        for slang, norm in self.phrase_dict.items():
            text = re.sub(r'\b' + re.escape(slang) + r'\b', norm, text, flags=re.IGNORECASE)
        return text

    def lexical_normalize(self, text):
        """Map các từ đơn còn sót lại"""
        words = text.split()
        normalized_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.word_dict:
                normalized_words.append(self.word_dict[lower_word])
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    def segment_words(self, text):
        """VŨ KHÍ MỚI: Cắt từ tiếng Việt bằng Underthesea"""
        # Tham số format="text" sẽ tự động nối các từ ghép bằng dấu gạch dưới (_)
        return word_tokenize(text, format="text")

    def run_pipeline(self, raw_text):
        if not isinstance(raw_text, str) or not raw_text.strip():
            return None
            
        text = self.normalize_unicode(raw_text)
        text = self.clean_text(text)
        text = self.phrase_normalize(text) # <--- Thêm bước này
        text = self.lexical_normalize(text)
        text = self.segment_words(text)
        return text

def fetch_and_preprocess():
    print("[INFO] BẮT ĐẦU KÉO VÀ TIỀN XỬ LÝ DỮ LIỆU...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # 1. Data Collection: Kéo log chat
        query = "SELECT id, session_id, raw_text, predicted_intent FROM ai_chat_analytics;"
        df = pd.read_sql_query(query, conn)
        print(f"📥 Đã kéo {len(df)} dòng dữ liệu thô từ Database.")
        
        # 2. Deduplication & Filtering
        # Lọc bỏ câu quá ngắn (< 3 ký tự)
        df = df[df['raw_text'].str.len() >= 3]
        
        # Tạo cột Hash MD5 và xóa trùng lặp (Exact Duplicate Removal)
        preprocessor = DataPreprocessor()
        df['text_hash'] = df['raw_text'].apply(preprocessor.generate_md5)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['text_hash'])
        print(f"✂️ Đã xóa {initial_count - len(df)} dòng trùng lặp hoàn toàn.")

        # 3. Unicode Normalization, Text Cleaning & Lexical Normalization
        print("[INFO] Đang chạy đường ống làm sạch (Phase A)...")
        df['cleaned_text'] = df['raw_text'].apply(preprocessor.run_pipeline)
        
        # In ra kết quả so sánh để đối chiếu
        print("\n" + "="*50)
        print("🔍 KẾT QUẢ SO SÁNH (TRƯỚC -> SAU):")
        print("="*50)
        for _, row in df.head(20).iterrows():
            print(f"Gốc  : {row['raw_text']}")
            print(f"Sạch : {row['cleaned_text']}")
            print("-" * 30)

        import os
        os.makedirs('data', exist_ok=True)
        # Bắt buộc phải cài thư viện openpyxl: pip install openpyxl
        df.to_csv('data/cleaned_chat_logs.csv', index=False, encoding='utf-8-sig')
        print("Đã lưu phiên bản chuẩn hóa ra file: data/cleaned_chat_logs.csv")

        # Trả về DataFrame đã làm sạch để dùng cho Phase B
        return df

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    clean_df = fetch_and_preprocess()