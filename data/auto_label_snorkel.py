import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
import warnings
import re
import yaml
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. ĐỊNH NGHĨA BỘ INTENT (14 fine-grained intents)
# ==========================================
ABSTAIN = -1
SEARCH_DESTINATION = 0
SEARCH_ACTIVITY = 1
SEARCH_PRICE = 2
ASK_TOUR_INFO = 3
SEARCH_ACCOMMODATION = 4
SEARCH_TRAVEL = 5
OUT_OF_SCOPE = 6
SEARCH_FOOD_DINING = 7
ASK_ITINERARY = 8
ASK_TRANSPORTATION = 9
ASK_POLICY_BOOKING = 10
ASK_WEATHER_TIMING = 11
ASK_LOCATION_FEATURE = 12
BOOK_TOUR = 13

INTENT_NAMES = {
    0: "search_destination", 1: "search_activity", 2: "search_price",
    3: "ask_tour_info", 4: "search_accommodation", 5: "search_travel",
    6: "out_of_scope", 7: "search_food_dining", 8: "ask_itinerary",
    9: "ask_transportation", 10: "ask_policy_booking", 11: "ask_weather_timing",
    12: "ask_location_feature", 13: "book_tour", -1: "UNLABELED"
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def has_keyword(text, keywords):
    """Check keyword with word boundary protection"""
    text_clean = str(text).lower().replace("_", " ")
    text_padded = f" {text_clean} "
    for kw in keywords:
        kw_clean = kw.replace("_", " ")
        if f" {kw_clean} " in text_padded:
            return True
    return False


def load_location_list():
    """Load destination list from Rasa lookup table"""
    try:
        path = "rasa_bot/data/train/nlu.yml"
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        doc = yaml.safe_load(content)
        for item in doc.get("nlu", []):
            if item.get("lookup") == "location":
                lines = item.get("examples", "").strip().split("\n")
                locations = []
                for loc in lines:
                    loc = loc.strip().lower()
                    if loc.startswith("- "):
                        loc = loc[2:]
                    if loc:
                        locations.append(loc)
                return locations
    except Exception as e:
        print(f"⚠️ Không load được location list: {e}")
        return None
    return None


DESTINATIONS = load_location_list() or [
    "hà nội", "hồ chí minh", "sài gòn", "đà nẵng", "hải phòng", "cần thơ",
    "đà lạt", "nha trang", "huế", "hội an", "vũng tàu", "phú quốc", "sapa",
    "sa pa", "hạ long", "hà giang", "cao bằng", "ninh bình", "phan thiết",
    "mũi né", "côn đảo", "phú yên", "quy nhơn", "tam đảo", "mộc châu",
    "mai châu", "pù luông", "sầm sơn", "cửa lò", "phan rang", "bình ba",
    "nam du", "thanh hóa", "nghệ an", "hà tĩnh", "quảng bình", "quảng nam",
    "bình định", "khánh hòa", "bình thuận", "gia lai", "đắk lắk", "lâm đồng",
    "tây ninh", "bình dương", "đồng nai", "bến tre"
]

print(f"📌 Đã load {len(DESTINATIONS)} địa danh cho labeling functions")


def has_destination(text):
    return has_keyword(text, DESTINATIONS)


# ==========================================
# 3. LABELING FUNCTIONS (cho phép overlaps)
# ==========================================

@labeling_function()
def lf_search_accommodation(x):
    keywords = ["đặt phòng", "khách sạn", "resort", "homestay", "chỗ ở", "villa",
                "thuê phòng", "book phòng", "nhà nghỉ", "khu nghỉ dưỡng",
                "chỗ nghỉ", "nghỉ dưỡng", "phòng", "hostel", "biệt thự"]
    return SEARCH_ACCOMMODATION if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_search_price(x):
    keywords_price = ["bao nhiêu tiền", "chi phí", "giá vé", "bảng giá",
                      "giá phòng", "giá tour", "tốn bao nhiêu", "giá rẻ",
                      "khuyến mãi", "thiệt hại", "giá combo", "giá trọn gói"]
    if has_keyword(x.cleaned_text, keywords_price):
        return SEARCH_PRICE
    text = str(x.cleaned_text).lower().replace("_", " ")
    if re.search(r'\b\d+\s*(triệu|củ|nghìn|ngàn|tr|k)\b', text):
        return SEARCH_PRICE
    return ABSTAIN


@labeling_function()
def lf_search_activity(x):
    keywords = ["leo núi", "trekking", "cắm trại", "săn mây", "lặn san hô",
                "lặn biển", "dù lượn", "cáp treo", "tắm biển", "đạp xe",
                "chèo thuyền", "kayak", "đi bộ", "hiking", "ngắm hoa",
                "chụp ảnh", "sống ảo", "tham quan", "câu cá", "trượt tuyết",
                "zipline", "chèo sup", "du thuyền", "cưỡi ngựa", "đi phượt",
                "road trip", "camping", "dã ngoại", "picnic", "ngắm hoàng hôn"]
    return SEARCH_ACTIVITY if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_search_destination(x):
    keywords = ["review", "đẹp", "nên đi", "gợi ý", "chỗ nào",
                "địa điểm", "check in", "kinh nghiệm", "đi đâu",
                "đến", "thăm", "ghé", "viếng"]
    if has_destination(x.cleaned_text):
        if has_keyword(x.cleaned_text, keywords):
            return SEARCH_DESTINATION
    return ABSTAIN


@labeling_function()
def lf_ask_tour_info(x):
    keywords = ["lịch trình", "thông tin tour", "bao gồm", "hướng dẫn viên",
                "khởi hành", "mấy giờ", "chi tiết tour", "báo giá tour",
                "tour bao gồm", "có bao ăn", "phụ thu", "lưu trú",
                "ăn uống", "tour này", "tour đó"]
    return ASK_TOUR_INFO if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_out_of_scope(x):
    positive_kw = ["giá vàng", "tổng thống", "nấu ăn", "mở nhạc",
                   "1 cộng 1", "bitcoin", "đề về", "câu chuyện cười",
                   "đội nào", "bài văn", "trồng cây", "chữa đau dạ dày",
                   "code python", "học bài", "bài tập", "giải toán"]
    if has_keyword(x.cleaned_text, positive_kw):
        return OUT_OF_SCOPE
    travel_kw = ["du lịch", "đi", "tour", "khách sạn", "đặt", "giá"]
    if has_keyword(x.cleaned_text, travel_kw):
        return ABSTAIN
    return ABSTAIN


@labeling_function()
def lf_search_travel(x):
    keywords = ["muốn đi du lịch", "đi chơi", "vi vu", "set kèo",
                "muốn đi đâu", "muốn vi vu", "đi đâu đó", "đi phượt",
                "muốn booking", "muốn đặt tour", "tìm tour", "đi du lịch",
                "du lịch", "muốn đi", "tôi muốn đi", "mình muốn đi"]
    return SEARCH_TRAVEL if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_search_food_dining(x):
    keywords = ["ăn gì", "đặc sản", "món ngon", "quán ăn", "nhà hàng",
                "ẩm thực", "hải sản", "ăn sáng", "ăn trưa", "ăn tối",
                "ăn vặt", "đồ ăn", "món", "quán nhậu", "cà phê",
                "món gì ngon", "địa chỉ ăn", "ăn nhức nách", "cực dính",
                "chân ái", "list đồ ăn", "đặc sản gì"]
    return SEARCH_FOOD_DINING if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_ask_itinerary(x):
    keywords = ["lịch trình", "plan", "kế hoạch", "sắp xếp lịch",
                "tự túc", "phân bổ thời gian", "nên đi những đâu",
                "tạo lịch trình", "lên plan", "lịch trình ăn chơi",
                "lên giúp", "thiết kế lịch", "kế hoạch du lịch"]
    return ASK_ITINERARY if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_ask_transportation(x):
    keywords = ["máy bay", "xe khách", "tàu hỏa", "xe máy", "taxi",
                "xe bus", "limousine", "phương tiện", "di chuyển",
                "vé máy bay", "tàu thủy", "xe đưa đón", "thuê xe",
                "xe giường nằm", "tàu cao tốc", "xe ôm", "grab",
                "xe đạp", "ô tô", "đi bằng", "bằng phương tiện"]
    return ASK_TRANSPORTATION if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_ask_policy_booking(x):
    keywords = ["hủy tour", "hoàn tiền", "chính sách", "đặt cọc",
                "thanh toán", "hóa đơn", "thủ tục", "visa",
                "hủy phòng", "đổi ngày", "phụ thu", "miễn phí",
                "trẻ em", "người cao tuổi", "xuất hóa đơn",
                "chuyển khoản", "ốm hủy", "bao lâu nhận", "điều kiện"]
    return ASK_POLICY_BOOKING if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_ask_weather_timing(x):
    if has_destination(x.cleaned_text):
        return ABSTAIN
    keywords = ["thời tiết", "mùa nào", "tháng mấy", "thời điểm",
                "mùa mưa", "mùa khô", "nên đi tháng", "đẹp nhất",
                "hợp lý nhất", "mùa này", "mùa đó"]
    return ASK_WEATHER_TIMING if has_keyword(x.cleaned_text, keywords) else ABSTAIN


@labeling_function()
def lf_ask_location_feature(x):
    keywords = ["thế nào", "ra sao", "có gì", "phong cảnh",
                "khí hậu", "mát không", "lạnh không", "có đẹp",
                "có ồn", "an ninh", "nhộn nhịp", "gì ở",
                "chơi gì", "có gì hay", "có gì chơi", "view",
                "vị trí", "thuận tiện", "gần biển"]
    if has_keyword(x.cleaned_text, keywords) and has_destination(x.cleaned_text):
        return ASK_LOCATION_FEATURE
    return ABSTAIN


@labeling_function()
def lf_book_tour(x):
    keywords = ["đặt tour", "book tour", "đặt chỗ", "muốn đặt",
                "đặt giúp", "book giúp", "muốn book", "đặt hộ"]
    return BOOK_TOUR if has_keyword(x.cleaned_text, keywords) else ABSTAIN


# ==========================================
# 4. CHẠY PIPELINE
# ==========================================

def run_auto_labeling():
    print("[INFO] Đang nạp dữ liệu vào hệ thống Snorkel...")
    try:
        df = pd.read_csv('data/cleaned_chat_logs.csv')
        df.columns = [c.strip() for c in df.columns]
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file cleaned_chat_logs.csv.")
        return

    df = df.dropna(subset=['cleaned_text'])
    print(f"📊 Tổng số rows: {len(df)}")

    print("🧠 Các chuyên gia đang đánh giá theo 14 nhóm Intent...")
    lfs = [
        lf_search_accommodation, lf_search_price, lf_search_activity,
        lf_search_destination, lf_ask_tour_info, lf_out_of_scope,
        lf_search_travel, lf_search_food_dining, lf_ask_itinerary,
        lf_ask_transportation, lf_ask_policy_booking, lf_ask_weather_timing,
        lf_ask_location_feature, lf_book_tour
    ]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)

    lf_names = [lf.name for lf in lfs]
    analysis = LFAnalysis(L_train, lfs).lf_summary()
    analysis.index = lf_names
    print("\n=== LF Analysis ===")
    print(analysis.to_string())

    cov = (L_train != -1).sum(axis=1)
    covered = (cov > 0).sum()
    multi = (cov > 1).sum()
    print(f"\n📈 Coverage: {covered}/{len(df)} ({covered/len(df)*100:.1f}%)")
    print(f"📈 Overlap rate (2+ LFs): {multi}/{len(df)} ({multi/len(df)*100:.1f}%)")

    print("⚖️ Label Model đang tính toán trọng số...")
    label_model = LabelModel(cardinality=14, verbose=False)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    preds, probs = label_model.predict(L=L_train, return_probs=True)

    df['snorkel_intent'] = [INTENT_NAMES.get(p, "UNLABELED") for p in preds]
    df['snorkel_confidence'] = probs.max(axis=1)

    df.to_csv('data/labeled_chat_logs.csv', index=False)

    print("\n✅ HOÀN TẤT GÁN NHÃN!")
    print("-" * 40)
    print("📊 BÁO CÁO SỐ LƯỢNG INTENT:")
    print(df['snorkel_intent'].value_counts().to_string())
    print(f"🔥 ĐỘ TỰ TIN TRUNG BÌNH: {df['snorkel_confidence'].mean():.2f}")
    print(f"📊 PHÂN PHỐI CONFIDENCE:")
    print(df['snorkel_confidence'].describe().to_string())


if __name__ == "__main__":
    run_auto_labeling()
