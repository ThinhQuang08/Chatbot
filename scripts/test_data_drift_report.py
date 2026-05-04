import pandas as pd
import warnings
from evidently import Report
from evidently.presets import DataDriftPreset

# Tắt các cảnh báo chia cho 0 do tập data nhỏ
warnings.filterwarnings('ignore')

print("[INFO] Bắt đầu quá trình giả lập 3 kịch bản Data Drift...")

# ==========================================
# 0. TẠO DỮ LIỆU CHUẨN (REFERENCE DATA)
# ==========================================
reference_data = pd.DataFrame({
    'intent': ['search_travel', 'search_travel', 'ask_location_feature', 'search_travel', 'ask_location_feature'] * 20,
    'destination': ['Đà Lạt', 'Vũng Tàu', 'Sapa', 'Đà Lạt', 'Phú Quốc'] * 20,
    'budget': [5000000, 3000000, 4000000, 4500000, 8000000] * 20,
    'category': ['khách sạn', 'resort', 'homestay', 'khách sạn', 'resort'] * 20,
    'month': [1, 2, 3, 1, 2] * 20  # Data gốc toàn đi mùa Xuân (Tháng 1, 2, 3)
})

# ==========================================
# KỊCH BẢN 1: LẠM PHÁT (BUDGET DRIFT)
# ==========================================
# Vẫn đi các nơi cũ, nhưng budget rớt thê thảm xuống dưới 1 triệu
current_data_scenario_1 = pd.DataFrame({
    'intent': ['search_travel', 'search_travel', 'ask_location_feature', 'search_travel', 'ask_location_feature'] * 20,
    'destination': ['Đà Lạt', 'Vũng Tàu', 'Sapa', 'Đà Lạt', 'Phú Quốc'] * 20,
    'budget': [800000, 500000, 1000000, 1200000, 900000] * 20, # <--- SỰ KHÁC BIỆT Ở ĐÂY
    'category': ['khách sạn', 'resort', 'homestay', 'khách sạn', 'resort'] * 20,
    'month': [1, 2, 3, 1, 2] * 20
})

# ==========================================
# KỊCH BẢN 2: THAY ĐỔI HÀNH VI (INTENT DRIFT)
# ==========================================
# Mùa mưa bão, khách toàn hỏi thời tiết, hủy vé (out_of_scope)
current_data_scenario_2 = pd.DataFrame({
    'intent': ['out_of_scope', 'out_of_scope', 'search_travel', 'out_of_scope', 'out_of_scope'] * 20, # <--- SỰ KHÁC BIỆT
    'destination': ['Đà Lạt', 'Vũng Tàu', 'Sapa', 'Đà Lạt', 'Phú Quốc'] * 20,
    'budget': [5000000, 3000000, 4000000, 4500000, 8000000] * 20,
    'category': ['khách sạn', 'resort', 'homestay', 'khách sạn', 'resort'] * 20,
    'month': [7, 8, 7, 8, 7] * 20
})

# ==========================================
# KỊCH BẢN 3: LỆCH MÙA (SEASON DRIFT)
# ==========================================
# Khách vẫn tìm Đà Lạt, Vũng Tàu, nhưng toàn tìm vào cuối năm (Tháng 10, 11, 12)
current_data_scenario_3 = pd.DataFrame({
    'intent': ['search_travel', 'search_travel', 'ask_location_feature', 'search_travel', 'ask_location_feature'] * 20,
    'destination': ['Đà Lạt', 'Vũng Tàu', 'Sapa', 'Đà Lạt', 'Phú Quốc'] * 20,
    'budget': [5000000, 3000000, 4000000, 4500000, 8000000] * 20,
    'category': ['khách sạn', 'resort', 'homestay', 'khách sạn', 'resort'] * 20,
    'month': [10, 11, 12, 10, 11] * 20 # <--- SỰ KHÁC BIỆT Ở ĐÂY
})

# ==========================================
# HÀM CHẠY REPORT
# ==========================================
def generate_report(ref_data, cur_data, filename):
    print(f"🔄 Đang tạo báo cáo: {filename}...")
    report = Report(metrics=[DataDriftPreset()])
    eval_result = report.run(reference_data=ref_data, current_data=cur_data)
    eval_result.save_html(filename)
    print(f"✅ Đã lưu: {filename}")

# Chạy cả 3 kịch bản
generate_report(reference_data, current_data_scenario_1, "report_1_budget_drift.html")
generate_report(reference_data, current_data_scenario_2, "report_2_intent_drift.html")
generate_report(reference_data, current_data_scenario_3, "report_3_season_drift.html")

print("\n🎉 HOÀN TẤT! Bạn hãy mở 3 file HTML vừa tạo để xem kết quả nhé!")