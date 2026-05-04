import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. ĐỊNH NGHĨA CÁC INTENT
# ==========================================
ABSTAIN = -1
SEARCH_TRAVEL = 0
ASK_PRICE = 1
BOOK_HOTEL = 2
OUT_OF_SCOPE = 3

# ==========================================
# 2. LABELING FUNCTIONS (ĐÃ NÂNG CẤP LOGIC LOẠI TRỪ)
# ==========================================

@labeling_function()
def lf_price(x):
    """Ưu tiên 1: Chỉ cần có từ khóa hỏi giá, tự tin dán nhãn ASK_PRICE"""
    keywords = ["giá", "nhiêu", "tiền", "chi_phí", "bao_nhiêu", "củ", "tr", "rẻ"]
    return ASK_PRICE if any(word in str(x.cleaned_text).lower() for word in keywords) else ABSTAIN

@labeling_function()
def lf_booking(x):
    """Ưu tiên 2: Nếu có từ khóa đặt chỗ, dán nhãn BOOK_HOTEL"""
    keywords = ["đặt", "book", "phòng", "khách_sạn", "resort", "thuê", "vé"]
    return BOOK_HOTEL if any(word in str(x.cleaned_text).lower() for word in keywords) else ABSTAIN

@labeling_function()
def lf_search_travel_strict(x):
    """
    Ưu tiên 3 (Strict Mode): Chỉ dán nhãn SEARCH_TRAVEL nếu có địa danh 
    NHƯNG KHÔNG CÓ từ khóa hỏi giá hay đặt phòng. Tránh cãi nhau!
    """
    text = str(x.cleaned_text).lower()
    destinations = ["đà_lạt", "nha_trang", "phú_quốc", "sapa", "sa_pa", "đà_nẵng", "hà_nội", "sài_gòn"]
    price_keywords = ["giá", "nhiêu", "tiền", "chi_phí", "củ", "rẻ"]
    book_keywords = ["đặt", "book", "phòng", "thuê", "vé"]
    
    has_dest = any(loc in text for loc in destinations)
    has_price = any(p in text for p in price_keywords)
    has_book = any(b in text for b in book_keywords)
    
    # Logic loại trừ xung đột
    if has_dest and not has_price and not has_book:
        return SEARCH_TRAVEL
    return ABSTAIN

@labeling_function()
def lf_out_of_scope(x):
    """Bắt các câu tào lao, chửi thề"""
    keywords = ["thời_tiết", "giải_toán", "nấu", "chửi", "ngu", "bài_tập", "tên_gì", "ăn_gì", "buồn_ngủ"]
    return OUT_OF_SCOPE if any(word in str(x.cleaned_text).lower() for word in keywords) else ABSTAIN

# ==========================================
# 3. CHẠY PIPELINE (TRAINING LABEL MODEL)
# ==========================================
def run_auto_labeling():
    print("[INFO] Đang nạp 1,500 dòng dữ liệu vào hệ thống Snorkel...")
    try:
        df = pd.read_csv('data/cleaned_chat_logs.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file. Bạn chạy file tiền xử lý trước nhé!")
        return
        
    df = df.dropna(subset=['cleaned_text'])

    print("🧠 Các chuyên gia (LFs) đang đọc và bỏ phiếu...")
    lfs = [lf_price, lf_booking, lf_search_travel_strict, lf_out_of_scope]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)

    print("⚖️ Label Model đang học quy luật phân phối từ 1,500 câu...")
    label_model = LabelModel(cardinality=4, verbose=False)
    # Fit model với 1500 dòng để tìm ra trọng số chính xác nhất cho từng LF
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    print("🎯 Đang xuất dự đoán cuối cùng...")
    preds, probs = label_model.predict(L=L_train, return_probs=True)
    
    intent_map = {0: "search_travel", 1: "ask_price", 2: "book_hotel", 3: "out_of_scope", -1: "UNLABELED"}
    df['snorkel_intent'] = [intent_map[p] for p in preds]
    df['snorkel_confidence'] = probs.max(axis=1)

    df.to_csv('data/labeled_chat_logs.csv', index=False)
    
    # ===== BÁO CÁO KIỂM ĐỊNH (DATA VALIDATION) =====
    print("\n✅ HOÀN TẤT GÁN NHÃN! Đã lưu file: data/labeled_chat_logs.csv")
    print("-" * 40)
    print("📊 BÁO CÁO SỐ LƯỢNG INTENT (CLASS BALANCE):")
    print(df['snorkel_intent'].value_counts())
    print("-" * 40)
    
    # In ra độ tự tin trung bình để xem mô hình đã "hết sợ" chưa
    mean_conf = df['snorkel_confidence'].mean()
    print(f"🔥 ĐỘ TỰ TIN TRUNG BÌNH (CONFIDENCE): {mean_conf:.2f}")
    if mean_conf >= 0.85:
        print("🎉 Tuyệt vời! Độ tự tin đã vượt ngưỡng an toàn (0.85). Dữ liệu này sẵn sàng để đưa vào luồng Auto-label.")
    else:
        print("⚠️ Độ tự tin vẫn dưới 0.85. Có thể cần bổ sung thêm luật LF hoặc xem lại các câu UNLABELED.")

if __name__ == "__main__":
    run_auto_labeling()