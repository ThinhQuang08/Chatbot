import re

class VietnameseNormalizer:
    def __init__(self):
        # 1. Từ điển viết tắt (Có thể load từ file JSON cho dễ quản lý)
        self.teencode_dict = {
            "ik": "đi",
            "đlạt": "Đà Lạt",
            "đl": "Đà Lạt",
            "wá": "quá",
            "z": "vậy",
            "k": "không",
            "ko": "không",
            "khong": "không",
            "dc": "được",
            "đc": "được",
            "j": "gì",
            "chjll": "chill",
            "mún": "muốn"
        }
        
    def _replace_teencode(self, text):
        """Thay thế teencode bằng từ chuẩn dựa trên từ điển"""
        words = text.split()
        normalized_words = []
        for word in words:
            # Chuyển về chữ thường để so khớp
            lower_word = word.lower()
            if lower_word in self.teencode_dict:
                normalized_words.append(self.teencode_dict[lower_word])
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    def _remove_repeated_characters(self, text):
        """Xóa các ký tự lặp lại quá 2 lần (ví dụ: đẹpppp -> đẹp)"""
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def normalize(self, raw_text):
        if not raw_text or not isinstance(raw_text, str):
            return ""
            
        text = raw_text.strip()
        
        # Bước 1: Giảm thiểu ký tự lặp
        text = self._remove_repeated_characters(text)
        
        # Bước 2: Mapping từ điển teencode
        text = self._replace_teencode(text)
        
        # Bước 3: Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        
        return text

# ==========================================
# TEST THỬ CHỨC NĂNG
# ==========================================
if __name__ == "__main__":
    normalizer = VietnameseNormalizer()
    
    test_sentences = [
        "mún ik đlạt wá z chời có kèo k",
        "pq có j chjll k bot",
        "tour đẹppppppp wáááááá"
    ]
    
    for sentence in test_sentences:
        clean_text = normalizer.normalize(sentence)
        print(f"Gốc:   {sentence}")
        print(f"Chuẩn: {clean_text}\n")
