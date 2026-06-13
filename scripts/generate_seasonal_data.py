# chạy 1 lần để tạo dữ liệu mùa phục vụ test drift
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml


def load_existing_texts(nlu_path):
    with open(nlu_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    texts = set()
    for item in data.get("nlu", []):
        examples = item.get("examples", "")
        for line in examples.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                texts.add(line[2:].strip().lower())
    return texts


WINTER_DATA = [
    ("đi Sa Pa ngắm tuyết", "search_destination"),
    ("thuê áo ấm ở Đà Lạt", "search_accommodation"),
    ("giá tour Đà Lạt tháng 12", "search_price"),
    ("thời tiết mùa đông Hà Nội", "ask_weather_timing"),
    ("homestay có lò sưởi ở Sa Pa", "search_accommodation"),
    ("săn mây Tà Xùa tháng 1", "search_activity"),
    ("check in hoa tam giác mạch Hà Giang", "search_destination"),
    ("ruộng bậc thang mùa lúa chín", "search_activity"),
    ("leo núi Bạch Mộc Lương Tử", "search_activity"),
    ("mang theo áo ấm đi Đà Lạt", "inform"),
    ("đi xe máy lên Sa Pa", "ask_transportation"),
    ("tour Hà Giang 4 ngày 3 đêm", "search_travel"),
    ("ngắm hoàng hôn trên đỉnh núi", "search_activity"),
    ("nhà nghỉ gần trạm xe ở Sa Pa", "search_accommodation"),
    ("giá vé máy bay đi Điện Biên", "ask_transportation"),
    ("đi chợ tình Sa Pa mùa đông", "search_destination"),
    ("ẩm thực phố Hà Nội mùa đông", "search_food_dining"),
    ("đặt phòng homestay ở Sa Pa", "book_tour"),
    ("tham quan thác Bạc Sa Pa", "search_activity"),
    ("tour mùa đông giá rẻ", "search_price"),
    ("nhà xe đi Sa Pa từ Hà Nội", "ask_transportation"),
    ("ở lại mấy ngày ở Sa Pa", "ask_itinerary"),
    ("cho thuê áo phao ở Đà Lạt", "search_activity"),
    ("cơm tấm ngon ở Hà Nội", "search_food_dining"),
    ("khách sạn gần trung tâm Sa Pa", "search_accommodation"),
    ("đi chợ ban ngày ở Hà Nội", "ask_itinerary"),
    ("địa điểm cắm trại gần Hà Nội", "search_destination"),
    ("có tuyến xe buýt đi Sa Pa không", "ask_transportation"),
    ("phòng nghỉ view núi ở Sa Pa", "search_accommodation"),
    ("giá tour Đà Lạt trọn gói", "search_price"),
    ("mang quần áo gì đi Đà Lạt tháng 11", "inform"),
    ("ngắm tuyết rơi ở Sa Pa", "search_activity"),
    ("nhà hàng ngon nhất Sa Pa", "search_food_dining"),
    ("đi Hà Nội mùa đông chơi gì", "ask_itinerary"),
    ("thuê xe máy đi thác Bạc", "search_activity"),
    ("địa điểm chụp ảnh đẹp ở Đà Lạt", "search_destination"),
    ("khách sạn giá rẻ ở Hà Nội", "search_accommodation"),
    ("tour Đà Lạt mùa đông cho gia đình", "search_travel"),
    ("thời tiết Hà Nội tháng 10", "ask_weather_timing"),
    ("đi Lào Cai bằng tàu hỏa", "ask_transportation"),
    ("cơm rang bò Hà Nội", "search_food_dining"),
    ("phượt xe máy vòng quanh Tây Bắc", "search_travel"),
    ("check in Nhà thờ Lớn Hà Nội", "search_destination"),
    ("vé máy bay đi Điện Biên giá rẻ", "ask_transportation"),
    ("homestay view đẹp nhất Sa Pa", "search_accommodation"),
    ("đi chợ mùa đông ở Hà Nội", "search_activity"),
    ("thuê áo ấm ở đâu tại Đà Lạt", "inform"),
    ("phòng đơn giá rẻ ở Sa Pa", "search_price"),
    ("ngắm mây trên đỉnh Fansipan", "search_activity"),
    ("bắt xe đi Sa Pa ở bến xe Mỹ Đình", "ask_transportation"),
    ("đặt tour Hà Giang từ Hà Nội", "search_travel"),
    ("lẩu cá hồi ở Sa Pa", "search_food_dining"),
    ("khách sạn gần ga Sa Pa", "search_accommodation"),
    ("địa điểm đi chơi đêm ở Hà Nội", "search_destination"),
    ("xe điện tham quan phố cổ Hà Nội", "ask_transportation"),
    ("đặt tour Đà Lạt 2 ngày 1 đêm", "book_tour"),
    ("đi Sa Pa tháng mấy đẹp nhất", "ask_weather_timing"),
    ("thuê xe máy đi Hà Giang", "search_activity"),
    ("cơm chay ngon ở Hà Nội", "search_food_dining"),
    ("nhà chơi trẻ em gần Hồ Tây", "search_destination"),
    ("tour Hà Giang giá sinh viên", "search_price"),
]

SUMMER_DATA = [
    ("tour lặn biển Nha Trang", "search_activity"),
    ("giá vé VinWonders Nam Hội An", "search_price"),
    ("đi Phú Quốc tháng 6", "search_destination"),
    ("resort view biển Vũng Tàu", "search_accommodation"),
    ("thuê canô đi đảo", "search_activity"),
    ("hải sản tươi sống ở Cát Bà", "search_food_dining"),
    ("vé máy bay từ Sài Gòn ra Đà Nẵng", "ask_transportation"),
    ("tour phượt miền Trung bằng xe máy", "search_travel"),
    ("ngắm hoàng hôn trên biển", "search_activity"),
    ("khách sạn gần biển ở Phú Quốc", "search_accommodation"),
    ("giá trò chơi nước ở VinWonders", "search_price"),
    ("đi đảo Nam Du mùa hè", "search_destination"),
    ("thuê máy ảnh dưới nước", "search_activity"),
    ("món nướng hải sản ngon ở Nha Trang", "search_food_dining"),
    ("đi Nha Trang tháng mấy đẹp", "ask_weather_timing"),
    ("tour Phú Quốc 3 ngày 2 đêm", "search_travel"),
    ("resort có hồ bơi vô cực", "search_accommodation"),
    ("lái xe máy vòng quanh đảo", "search_activity"),
    ("đặt tour Nha Trang giá rẻ", "book_tour"),
    ("hải sản đêm ở Vũng Tàu", "search_food_dining"),
    ("cho thuê xuồng đi đảo Cát Bà", "search_activity"),
    ("giá taxi từ sân bay Đà Nẵng vào trung tâm", "ask_transportation"),
    ("nhà nghỉ gần biển Nha Trang", "search_accommodation"),
    ("đi biển mùa hè ở miền Trung", "search_destination"),
    ("thuê canô đi câu cá", "search_activity"),
    ("cơm niêu ở Đà Nẵng", "search_food_dining"),
    ("tour Đà Nẵng Hội An 5 ngày", "search_travel"),
    ("bungalow view biển giá rẻ", "search_accommodation"),
    ("vé tàu hỏa đi Nha Trang", "ask_transportation"),
    ("đi Phú Quốc từ Hà Nội", "ask_transportation"),
    ("thèm đi biển quá", "inform"),
    ("ẩm thực biển ở Phan Thiết", "search_food_dining"),
    ("homestay gần biển ở Đà Nẵng", "search_accommodation"),
    ("lặn biển ngắm san hô", "search_activity"),
    ("giá phòng resort ở Nha Trang", "search_price"),
    ("đi Cát Bà tháng 7", "search_destination"),
    ("vé tàu cao tốc ra đảo", "ask_transportation"),
    ("quán ốc ngon ở Sài Gòn", "search_food_dining"),
    ("tour phượt Phú Quốc bằng xe máy", "search_travel"),
    ("khách sạn có lối đi ra biển riêng", "search_accommodation"),
    ("đi chơi Vũng Tàu trong ngày", "ask_itinerary"),
    ("nước mắm ngon mua về làm quà", "search_food_dining"),
    ("thuê căn gia đình ở Vinpearl", "search_price"),
    ("đi đảo ở Nha Trang", "search_activity"),
    ("đặt vé máy bay đi Phú Quốc", "book_tour"),
    ("phòng Airbnb Đà Nẵng view biển", "search_accommodation"),
    ("thời tiết Nha Trang tháng 8", "ask_weather_timing"),
    ("lẩu cá bớp ở Sài Gòn", "search_food_dining"),
    ("cho thuê đồ lặn biển", "search_activity"),
    ("cơm tấm ngon ở Đà Nẵng", "search_food_dining"),
    ("đi Phú Quốc chơi gì thú vị", "ask_itinerary"),
    ("resort 5 sao ở miền Trung", "search_accommodation"),
    ("tour du thuyền Nha Trang", "search_travel"),
    ("giá vé VinWonders Nha Trang", "search_price"),
    ("đi biển gần Sài Gòn chỗ nào đẹp", "search_destination"),
    ("thuê xe 7 chỗ ở Nha Trang đi đảo", "ask_transportation"),
    ("khách sạn có spa gần biển", "search_accommodation"),
    ("mua gì mang về từ Phú Quốc", "inform"),
    ("đi biển Đà Nẵng tháng 6", "search_activity"),
]


def write_csv(path, data):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        for row in data:
            writer.writerow(row)


def run():
    nlu_path = os.path.join(os.path.dirname(__file__), "..", "rasa_bot/data/train/nlu.yml")
    existing = load_existing_texts(nlu_path)
    print(f"Da doc {len(existing)} existing examples tu nlu.yml")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    winter_unique = [(t, i) for t, i in WINTER_DATA if t.lower() not in existing]
    summer_unique = [(t, i) for t, i in SUMMER_DATA if t.lower() not in existing]

    removed_winter = len(WINTER_DATA) - len(winter_unique)
    removed_summer = len(SUMMER_DATA) - len(summer_unique)
    if removed_winter:
        print(f"Da loai bo {removed_winter} winter examples bi trung voi nlu.yml")
    if removed_summer:
        print(f"Da loai bo {removed_summer} summer examples bi trung voi nlu.yml")

    winter_path = os.path.join(out_dir, "reference_winter.csv")
    summer_path = os.path.join(out_dir, "current_summer.csv")
    write_csv(winter_path, winter_unique)
    write_csv(summer_path, summer_unique)

    print(f"Da tao winter data: {len(winter_unique)} examples -> {winter_path}")
    print(f"Da tao summer data: {len(summer_unique)} examples -> {summer_path}")


if __name__ == "__main__":
    run()
