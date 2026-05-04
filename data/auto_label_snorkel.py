# data/auto_label_snorkel.py
# B. Gán nhãn tự động

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
# 2. LABELING FUNCTIONS (CHIẾN THUẬT OVERLAPPING)
# ==========================================

@labeling_function()
def lf_price(x):
    keywords = ["giá", "nhiêu", "tiền", "chi_phí", "bao_nhiêu", "củ", "tr", "rẻ"]
    return ASK_PRICE if any(word in str(x.cleaned_text).lower() for word in keywords) else ABSTAIN

@labeling_function()
def lf_booking(x):
    keywords = ["đặt", "book", "phòng", "khách_sạn", "resort", "thuê", "vé"]
    return BOOK_HOTEL if any(word in str(x.cleaned_text).lower() for word in keywords) else ABSTAIN

# --- TÁCH SEARCH_TRAVEL THÀNH 3 CHUYÊN GIA KHÁC NHAU ---

@labeling_function()
def lf_travel_destination(x):
    """Chuyên gia 1: Nhìn vào địa danh"""
    text = str(x.cleaned_text).lower()
    destinations = ["đà_lạt", "nha_trang", "phú_quốc", "sapa", "sa_pa", "đà_nẵng", "hà_nội", "sài_gòn"]
    has_dest = any(loc in text for loc in destinations)
    
    # Tránh xung đột với Giá / Đặt phòng
    if has_dest and lf_price(x) == ABSTAIN and lf_booking(x) == ABSTAIN:
        return SEARCH_TRAVEL
    return ABSTAIN

@labeling_function()
def lf_travel_action(x):
    """Chuyên gia 2: Nhìn vào động từ (đi, chơi, tour)"""
    text = str(x.cleaned_text).lower()
    actions = ["tour", "đi", "ik", "chơi", "du_lịch", "tham_quan"]
    has_action = any(act in text for act in actions)
    
    if has_action and lf_price(x) == ABSTAIN and lf_booking(x) == ABSTAIN:
        return SEARCH_TRAVEL
    return ABSTAIN

@labeling_function()
def lf_travel_intent_keywords(x):
    """Chuyên gia 3: Nhìn vào từ khóa chỉ mục đích tìm kiếm"""
    text = str(x.cleaned_text).lower()
    intents = ["tìm", "xem", "review", "kinh_nghiệm", "tư_vấn", "gợi_ý"]
    has_intent = any(i in text for i in intents)
    
    if has_intent and lf_price(x) == ABSTAIN and lf_booking(x) == ABSTAIN:
        return SEARCH_TRAVEL
    return ABSTAIN

@labeling_function()
def lf_out_of_scope(x):
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
    # Cập nhật danh sách LFs ở đây:
    lfs = [lf_price, lf_booking, lf_travel_destination, lf_travel_action, lf_travel_intent_keywords, lf_out_of_scope]
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