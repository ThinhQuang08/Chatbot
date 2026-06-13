"""Regenerate reference_normal_v3.csv with all 11 intents evenly distributed."""
import csv, random, os

random.seed(42)

PREFIXES = ["", "cho tôi hỏi ", "ad ơi ", "cho em hỏi ", "mình hỏi ", "làm ơn "]

destinations = ["Đà Lạt", "Nha Trang", "Phú Quốc", "Sa Pa", "Đà Nẵng", "Vũng Tàu", "Hà Nội", "Hội An", "Huế", "Mũi Né"]

templates_pool = {
    "search_travel": [
        "Tour {d} 3 ngày 2 đêm", "Tour {d} 4 ngày 3 đêm", "Tour {d} 5 ngày 4 đêm",
        "Tour du lịch {d} giá rẻ", "Combo {d} trọn gói", "Du lịch {d} tự túc",
        "Tour {d} cho gia đình", "Tour {d} nghỉ dưỡng", "Lịch trình tour {d}",
        "Tour {d} khởi hành từ Hà Nội", "Giá tour {d} 3 ngày", "Tour {d} giá sinh viên",
        "Tour {d} cuối tuần", "Tour {d} cho người già", "Booking tour {d} giá tốt",
        "Tour {d} 2 ngày 1 đêm", "Tour {d} 6 ngày 5 đêm", "Tour {d} trọn gói bao gồm",
        "Du lịch {d} mùa hè", "Du lịch {d} mùa đông", "Kỳ nghỉ {d}",
        "Tour {d} VIP 5 sao", "Tour {d} tiết kiệm", "Tour {d} khuyến mãi",
        "Tour {d} cho cặp đôi", "Tour {d} team building", "Du lịch {d} 1 ngày", "Tour {d} giá học sinh",
    ],
    "search_destination": [
        "Ở đâu chơi ở {d}", "Đi {d} mùa nào đẹp", "Có gì đẹp ở {d}",
        "Kinh nghiệm đi {d} lần đầu", "Review du lịch {d}", "Check in {d} ở đâu",
        "Đi {d} tham quan gì", "Cảnh đẹp {d} nổi tiếng", "Đến {d} nên đi đâu",
        "{d} có gì chơi", "Top địa điểm {d}", "Khám phá {d}",
        "Vẻ đẹp {d}", "Du lịch {d} có gì hay", "{d} ở đâu",
        "{d} cách Hà Nội bao xa", "{d} có gì đặc biệt", "Điểm đến {d}",
        "Nên đi {d} không", "{d} mùa nào đẹp nhất", "Giới thiệu về {d}",
        "Thông tin du lịch {d}", "{d} nổi tiếng về gì", "Bài viết về {d}",
        "Chia sẻ kinh nghiệm {d}", "{d} có đáng đi không",
    ],
    "search_accommodation": [
        "Khách sạn {d} giá rẻ", "Homestay đẹp ở {d}", "Resort {d} view biển",
        "Nhà nghỉ {d} trung tâm", "Khách sạn {d} gần biển", "Homestay {d} view đẹp",
        "Resort {d} 5 sao", "Nhà nghỉ {d} giá rẻ", "Khách sạn {d} trung tâm",
        "Lưu trú ở {d} gần chợ", "Villa {d} cho thuê", "Căn hộ {d} view đẹp",
        "Khu nghỉ dưỡng {d}", "Homestay {d} gần trung tâm", "Khách sạn {d} giá tốt",
        "Resort {d} hồ bơi", "Khách sạn {d} gần sân bay", "Nhà nghỉ {d} gần biển",
        "Homestay {d} cho đôi người", "Villa {d} hồ bơi", "Khách sạn {d} gần chợ đêm",
        "Resort {d} gia đình", "Khu nghỉ {d} view núi", "Camping {d} qua đêm",
        "Glamping {d}", "{d} có homestay nào đẹp", "Khách sạn {d} mới xây",
    ],
    "search_activity": [
        "Đi thác {d} tham quan", "Lặn biển ở {d}", "Đi chợ {d} về đêm",
        "Leo núi ở {d}", "Đi phượt {d} bằng xe máy", "Tắm biển {d}",
        "Tham quan phố cổ {d}", "Câu cá ở {d}", "Chèo thuyền ở {d}",
        "Đi bộ đường dài {d}", "Chơi dù bay {d}", "Lặn ngắm san hô {d}",
        "Chèo kayak {d}", "Đi cáp treo {d}", "Tham quan làng nghề {d}",
        "Đi chợ nổi {d}", "Xem biểu diễn {d}", "Khám phá hang động {d}",
        "Tắm nước nóng {d}", "Đi thuyền {d}", "Ăn tối trên du thuyền {d}",
        "Cắm trại {d}", "Đi bộ biển {d}", "Ngắm hoàng hôn {d}",
        "Chụp ảnh ở {d}", "Khám phá ẩm thực đường phố {d}",
    ],
    "search_food_dining": [
        "{d} có gì ngon", "Ẩm thực {d} món gì đặc sản", "Quán ăn ngon ở {d}",
        "Đặc sản {d} phải thử", "Hải sản {d} tươi ngon", "Ở {d} ăn gì ngon",
        "Món ngon {d} giá rẻ", "Chợ {d} ẩm thực", "Cafe view đẹp {d}",
        "Nhà hàng {d} sang trọng", "{d} có món gì đặc biệt", "Ăn vặt {d}",
        "Buffet {d} giá rẻ", "Quán nhậu {d}", "Món Việt {d} ngon",
        "{d} ăn gì cho bữa sáng", "Đồ nướng {d}", "{d} có quán nào nổi tiếng",
        "Lẩu {d} ăn ở đâu", "Chè {d} ngon", "Bánh xèo {d}",
        "Ăn chay {d}", "Hải sản {d} giá rẻ", "Mì quảng {d}",
    ],
    "search_price": [
        "Giá tour {d} bao nhiêu", "Chi phí du lịch {d} 3 ngày", "Vé máy bay đi {d} giá rẻ",
        "Giá khách sạn {d} 1 đêm", "Bảng giá tour {d}", "Kinh phí đi {d} tiết kiệm",
        "Giá homestay {d}", "Chi phí ăn uống ở {d}", "Giá vé tham quan {d}",
        "Tổng chi phí đi {d}", "Tour {d} giá bao nhiêu tiền", "Vé máy bay {d} khứ hồi",
        "Chi phí {d} 4 ngày 3 đêm", "Giá combo {d}", "Bảng báo giá {d}",
        "Giá dịch vụ {d}", "Chi phí thuê xe {d}", "Tiền ăn ở {d} 1 ngày",
        "Giá vé {d} cho trẻ em", "Chi phí đi {d} tự túc", "Ngân sách đi {d}",
        "Giá tham quan {d} 2026", "Khuyến mãi tour {d}", "Chi phí lưu trú {d}",
    ],
    "ask_transportation": [
        "Vé máy bay đi {d}", "Xe khách đi {d} từ Sài Gòn", "Tàu hỏa đi {d}",
        "Đi {d} bằng xe máy", "Từ Hà Nội đi {d} bằng gì", "Xe limousine đi {d}",
        "Phương tiện đi {d} rẻ nhất", "Từ Sài Gòn đi {d} bao xa", "Đường đi {d} có khó không",
        "Bay đi {d} từ Hà Nội", "Xe khách đi {d} giờ nào", "Tàu hỏa {d} bao nhiêu tiền",
        "Đi {d} bằng máy bay giá rẻ", "Xe giường nằm đi {d}", "Taxi ở {d} giá rẻ",
        "Thuê xe tự lái {d}", "Đi {d} bằng tàu cao tốc", "Xe buýt đi {d}",
        "Xe đạp đi {d}", "Từ {d} đi các nơi khác", "Các hãng bay đến {d}",
    ],
    "ask_weather_timing": [
        "Thời tiết {d} tháng 3", "Thời tiết {d} tháng 6", "Nên đi {d} tháng mấy",
        "Mùa mưa ở {d} tháng nào", "Nhiệt độ {d} trung bình", "Thời tiết {d} tháng 10",
        "Mùa khô ở {d}", "Mùa đẹp nhất ở {d}", "Đi {d} tháng 4 có mưa không",
        "Khí hậu {d} như thế nào", "Mùa bão {d}", "Thời tiết {d} tháng 12",
        "Thời tiết {d} tháng 2", "Nhiệt độ {d} tháng 7", "Đi {d} dịp lễ có vui không",
        "Thời tiết {d} cuối tuần", "Dự báo thời tiết {d}", "Lượng mưa {d} hàng năm",
        "Thời gian lý tưởng đi {d}", "{d} mùa xuân", "{d} mùa thu",
    ],
    "ask_itinerary": [
        "Ở {d} mấy ngày là đủ", "Lịch trình {d} 3 ngày 2 đêm", "Nên ở {d} bao lâu",
        "Gợi ý lịch trình {d}", "Đi {d} trong 2 ngày", "Lịch trình {d} tiết kiệm",
        "Ở {d} 4 ngày đi đâu", "Lịch trình {d} cho gia đình", "Đi {d} cuối tuần",
        "Lịch trình {d} tự túc", "Nên đi {d} mấy ngày", "Lịch trình {d} 5 ngày",
        "Gợi ý tour {d} 2 ngày", "Đi {d} 1 ngày thì đi đâu", "Lịch trình {d} nghỉ dưỡng",
        "Lịch trình {d} cho người già", "Đi {d} bao nhiêu ngày", "Kế hoạch du lịch {d}",
        "Lịch trình {d} sinh viên", "Lịch {d} phù hợp", "Đi {d} trong bao lâu",
    ],
    "book_tour": [
        "Đặt tour {d} online", "Đặt phòng khách sạn {d}", "Đặt homestay {d}",
        "Booking tour {d} giá rẻ", "Đặt vé máy bay đi {d}", "Đặt tour {d} trả góp",
        "Book phòng {d} view biển", "Đặt tour {d} khuyến mãi", "Mua tour {d} giảm giá",
        "Reserve phòng {d}", "Order tour {d}", "Đặt trước tour {d}",
        "Book combo {d}", "Giữ chỗ tour {d}", "Đặt tour {d} tận nơi",
        "Đặt phòng {d} online", "Booking homestay {d} giá rẻ", "Đặt tour {d} cho đoàn",
        "Mua vé tham quan {d}", "Reserve phòng {d} qua app", "Đặt tour {d} tiết kiệm",
    ],
    "inform": [
        "Tôi thích đi {d} vào mùa hè", "Gia đình tôi muốn đi {d} tuần tới",
        "Tôi đã đi {d} năm ngoái rất thích", "Nhóm bạn tôi định đi {d} tháng sau",
        "Tôi nghe nói {d} rất đẹp", "Tôi muốn tìm hiểu về {d}",
        "Vợ chồng tôi cưới ở {d}", "Công ty tôi tổ chức team building ở {d}",
        "Tôi ở {d} rồi muốn đi tiếp", "Sinh nhật tôi ở {d}",
        "Tôi thích {d} vì không khí mát mẻ", "Tôi muốn đưa con đi {d}",
        "Tôi đã đến {d} 3 lần rồi", "Bố mẹ tôi thích {d}",
        "Cả nhà tôi đi {d} dịp Tết", "Tôi mới đến {d} lần đầu",
        "Tôi muốn khám phá {d}", "Tôi yêu thích {d}",
        "Bạn tôi gợi ý đi {d}", "Tôi đã book tour {d} rồi",
    ],
}


def expand_with_prefixes(texts_with_intent):
    expanded = []
    for t, intent in texts_with_intent:
        for p in PREFIXES:
            expanded.append((p + t, intent))
    return expanded


# Build ref: evenly sample ~182 per destination to get ~2000 unique
unique_ref = []
for dest in destinations:
    # Collect all templates for this destination
    all_templates = []
    for intent, tmpls in templates_pool.items():
        for t in tmpls:
            all_templates.append((t.format(d=dest), intent))
    # Shuffle to mix intents
    random.shuffle(all_templates)
    # Take 27 per destination (before prefix expansion)
    selected = all_templates[:27]
    # Expand with prefixes → 27 * 6 = 162
    expanded = expand_with_prefixes(selected)
    random.shuffle(expanded)
    # Take up to 200
    unique_ref.extend(expanded[:200])

# Deduplicate
seen = set()
deduped = []
for t, intent in unique_ref:
    if t not in seen:
        seen.add(t)
        deduped.append((t, intent))

random.shuffle(deduped)
print(f"Ref unique before dup: {len(deduped)}")

N_TOTAL = 3000
N_DUP = N_TOTAL - len(deduped)
rows_ref = list(deduped)
if N_DUP > 0:
    rows_ref.extend(random.choices(deduped, k=N_DUP))
random.shuffle(rows_ref)

base = os.path.join(os.path.dirname(__file__) or ".", "..")
with open(os.path.join(base, "data", "reference_normal_v3.csv"), "w", encoding="utf-8") as f:
    f.write("text,intent\n")
    for text, intent in rows_ref:
        f.write(f"{text},{intent}\n")

# Quick verification
intent_counts = {}
for _, intent in rows_ref:
    intent_counts[intent] = intent_counts.get(intent, 0) + 1
print(f"Ref: {len(rows_ref)} rows, {len(deduped)} unique, {len(intent_counts)} intents")
for intent, count in sorted(intent_counts.items()):
    print(f"  {intent}: {count}")
print("Done!")
