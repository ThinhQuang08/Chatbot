import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import warnings
import re

warnings.filterwarnings('ignore')

# ==========================================
# 1. ĐỊNH NGHĨA BỘ INTENT MỚI (Fine-grained Intents)
# ==========================================
ABSTAIN = -1
SEARCH_DESTINATION = 0
SEARCH_ACTIVITY = 1
SEARCH_PRICE = 2
ASK_TOUR_INFO = 3
SEARCH_ACCOMMODATION = 4
SEARCH_TRAVEL = 5
OUT_OF_SCOPE = 6 # Vẫn phải giữ cái này để lọc rác nhé!

# ==========================================
# 2. CÁC CHUYÊN GIA BẮT INTENT CHUYÊN SÂU
# ==========================================

def has_keyword(text, keywords):
    """Xóa gạch dưới và bắt chính xác cụm từ, tránh lỗi Substring"""
    # Xóa gạch dưới do Underthesea tạo ra
    text_clean = str(text).lower().replace("_", " ")
    # Thêm khoảng trắng ở 2 đầu để làm "hàng rào" bảo vệ từ
    text_padded = f" {text_clean} " 
    
    for kw in keywords:
        kw_clean = kw.replace("_", " ")
        # Chỉ match khi từ đó đứng độc lập
        if f" {kw_clean} " in text_padded:
            return True
    return False

@labeling_function()
def lf_search_accommodation(x):
    keywords = ["đặt phòng", "phòng", "khách sạn", "resort", "homestay", "chỗ ở", "villa", "thuê phòng", "book phòng", "thuê resort"]
    return SEARCH_ACCOMMODATION if has_keyword(x.cleaned_text, keywords) else ABSTAIN

@labeling_function()
def lf_search_price(x):
    if lf_search_accommodation(x) == SEARCH_ACCOMMODATION:
        return ABSTAIN
        
    keywords = ["giá", "nhiêu", "tiền", "chi phí", "bao nhiêu", "củ", "tr", "rẻ", "khuyến mãi", "bảng giá"]
    return SEARCH_PRICE if has_keyword(x.cleaned_text, keywords) else ABSTAIN

@labeling_function()
def lf_price_voter(x):
    text = str(x.cleaned_text).lower().replace("_", " ")
    # Dùng regex có \b (word boundary) để bắt chính xác số đi kèm tiền
    if re.search(r'\b\d+\s*(triệu|củ|nghìn|tr|k)\b', text):
        return SEARCH_PRICE
    return ABSTAIN

@labeling_function()
def lf_search_activity(x):
    keywords = ["chơi", "trải nghiệm", "lặn", "trekking", "cắm trại", "hoạt động", "leo núi", "đạp xe", "xe đạp", "thuê xe", "dù lượn"]
    return SEARCH_ACTIVITY if has_keyword(x.cleaned_text, keywords) else ABSTAIN

@labeling_function()
def lf_search_destination(x):
    destinations = ["đà lạt", "nha trang", "phú quốc", "sapa", "sa pa", "đà nẵng", "hà nội", "sài gòn"]
    has_dest = has_keyword(x.cleaned_text, destinations)
    
    keywords = ["review", "thời tiết", "đặc sản", "kinh nghiệm", "chỗ nào", "mùa nào", "check in", "đẹp"]
    has_kw = has_keyword(x.cleaned_text, keywords)
    
    if has_dest and has_kw:
        return SEARCH_DESTINATION
    return ABSTAIN

@labeling_function()
def lf_ask_tour_info(x):
    keywords = ["lịch trình", "mấy ngày", "bao gồm", "thông tin", "di chuyển", "hướng dẫn viên", "phương tiện", "ăn uống", "chi tiết"]
    return ASK_TOUR_INFO if has_keyword(x.cleaned_text, keywords) else ABSTAIN

@labeling_function()
def lf_out_of_scope(x):
    text = str(x.cleaned_text).lower().replace("_", " ")
    destinations = ["đà lạt", "nha trang", "phú quốc", "sapa", "sa pa", "đà nẵng", "hà nội", "sài gòn"]
    has_dest = has_keyword(x.cleaned_text, destinations)
    
    if "thời tiết" in text and not has_dest:
        return OUT_OF_SCOPE
        
    keywords = ["giải toán", "nấu", "chửi", "ngu", "bài tập", "tên gì", "ăn gì", "buồn ngủ", "hôm nay"]
    return OUT_OF_SCOPE if has_keyword(x.cleaned_text, keywords) else ABSTAIN

@labeling_function()
def lf_search_travel(x):
    travel_kws = ["đi", "tour", "du lịch", "muốn đến", "vé"]
    destinations = ["đà lạt", "nha trang", "phú quốc", "sapa", "sa pa", "đà nẵng", "hà nội", "sài gòn"]
    
    has_travel = has_keyword(x.cleaned_text, travel_kws) or has_keyword(x.cleaned_text, destinations)
    
    is_specific = (
        lf_search_destination(x) != ABSTAIN or
        lf_search_activity(x) != ABSTAIN or
        lf_search_price(x) != ABSTAIN or
        lf_price_voter(x) != ABSTAIN or
        lf_ask_tour_info(x) != ABSTAIN or
        lf_search_accommodation(x) != ABSTAIN or
        lf_out_of_scope(x) != ABSTAIN
    )
    
    if has_travel and not is_specific:
        return SEARCH_TRAVEL
    return ABSTAIN

# ==========================================
# 4. CHẠY PIPELINE (TRAINING LABEL MODEL)
# ==========================================
def run_auto_labeling():
    print("[INFO] Đang nạp dữ liệu vào hệ thống Snorkel với BỘ INTENT MỚI...")
    try:
        df = pd.read_csv('data/cleaned_chat_logs.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file cleaned_chat_logs.csv.")
        return
        
    df = df.dropna(subset=['cleaned_text'])

    print("🧠 Các chuyên gia đang đánh giá theo 7 nhóm Intent...")
    lfs = [
        lf_search_destination, lf_search_activity, lf_search_price, 
        lf_ask_tour_info, lf_search_accommodation, lf_out_of_scope, 
        lf_search_travel # Để hàm tổng quát ở cuối cùng
    ]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)

    # Chú ý: cardinality giờ là 7 vì chúng ta có 7 class (từ 0 đến 6)
    print("⚖️ Label Model đang tính toán trọng số...")
    label_model = LabelModel(cardinality=7, verbose=False)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    preds, probs = label_model.predict(L=L_train, return_probs=True)
    
    # Map ID số nguyên về lại Text Intent
    intent_map = {
        0: "search_destination", 1: "search_activity", 2: "search_price", 
        3: "ask_tour_info", 4: "search_accommodation", 5: "search_travel",
        6: "out_of_scope", -1: "UNLABELED"
    }
    
    df['snorkel_intent'] = [intent_map[p] for p in preds]
    df['snorkel_confidence'] = probs.max(axis=1)

    df.to_csv('data/labeled_chat_logs.csv', index=False)
    
    print("\n✅ HOÀN TẤT GÁN NHÃN THEO BỘ INTENT MỚI!")
    print("-" * 40)
    print("📊 BÁO CÁO SỐ LƯỢNG INTENT:")
    print(df['snorkel_intent'].value_counts())
    print(f"🔥 ĐỘ TỰ TIN TRUNG BÌNH (CONFIDENCE): {df['snorkel_confidence'].mean():.2f}")

if __name__ == "__main__":
    run_auto_labeling()