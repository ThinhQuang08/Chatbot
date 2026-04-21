# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


import sys
import os
import re
from datetime import datetime
from typing import Optional, Tuple, Any, Text, Dict, List

from rasa_sdk.events import SlotSet
from rasa_sdk.forms import FormValidationAction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from services.search_service import search_destinations
from services.ai_response_service import generate_grounded_ai_response

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from database.db_connection import get_connection


SEASON_TO_MONTHS = {
    "mùa xuân": (1, 3), "mua xuan": (1, 3),
    "mùa hè": (4, 6), "mua he": (4, 6),
    "mùa thu": (7, 9), "mua thu": (7, 9),
    "mùa đông": (10, 12), "mua dong": (10, 12),
}


def parse_budget_vnd(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None

    text = str(value).lower().strip()

    unit_match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(triệu|trieu|tr|k|nghìn|ngàn|vnd|đ|dong)?",
        text,
    )

    if unit_match:
        value = float(unit_match.group(1).replace(",", "."))
        unit = unit_match.group(2)

        if unit in {"triệu", "trieu", "tr"}:
            return int(value * 1_000_000)
        if unit in {"k", "nghìn", "ngàn"}:
            return int(value * 1_000)
        if unit in {"vnd", "đ", "dong"}:
            return int(value)
        if value < 1000:
            return int(value * 1_000_000)
        return int(value)

    return None

def parse_month_range(
    season: Optional[str],
    month: Optional[object],
    month_from: Optional[object],
    month_to: Optional[object],
    time_window: Optional[str],
) -> Tuple[Optional[int], Optional[int]]:
    def extract_month(value: Optional[object]) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if 1 <= parsed <= 12 else None

        text = str(value).lower().strip()
        match = re.search(r"(1[0-2]|[1-9])", text)
        if not match:
            return None

        parsed = int(match.group(1))
        return parsed if 1 <= parsed <= 12 else None

    parsed_month_from = extract_month(month_from)
    parsed_month_to = extract_month(month_to)

    if parsed_month_from is not None and parsed_month_to is not None:
        return parsed_month_from, parsed_month_to

    parsed_month = extract_month(month)
    if parsed_month is not None:
        return parsed_month, parsed_month

    if season:
        season_value = str(season).lower().strip()
        if season_value in SEASON_TO_MONTHS:
            return SEASON_TO_MONTHS[season_value]

    if time_window and str(time_window).lower().strip() in {
        "sắp tới",
        "sap toi",
        "upcoming",
    }:
        current_month = datetime.now().month
        end_month = ((current_month + 2 - 1) % 12) + 1
        return current_month, end_month

    return None, None

class ValidateTravelForm(FormValidationAction):
    def name(self) -> Text:
        # Tên này bắt buộc phải có tiền tố validate_ + tên form của bạn
        return "validate_travel_form"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:

        # 1. LẤY DỮ LIỆU ĐỂ IN RA MÀN HÌNH TERMINAL
        user_message = tracker.latest_message.get("text", "")
        intent = tracker.latest_message.get("intent", {}).get("name")
        entities = tracker.latest_message.get("entities", [])
        
        print("\n[FORM VALIDATION LOG] " + "="*30, flush=True)
        print(f"USER NÓI: '{user_message}'")
        print(f"INTENT CHÍNH: {intent}")
        
        print("ENTITIES VỪA BẮT ĐƯỢC:")
        if not entities:
            print("   -> [Trống]")
        else:
            for e in entities:
                print(f"   -> {e.get('entity')}: '{e.get('value')}' ({e.get('extractor')})")
                
        print("TRÍ NHỚ (SLOTS) ĐANG GIỮ:")
        active_slots = {k: v for k, v in tracker.slots.items() if v is not None}
        if not active_slots:
            print("   -> [Trống]")
        else:
            for k, v in active_slots.items():
                print(f"   -> {k}: '{v}'")
        print("="*52 + "\n")

        # 2. TRẢ LẠI QUYỀN CHO RASA FORM XỬ LÝ TIẾP
        return await super().run(dispatcher, tracker, domain)

    def validate_budget(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Kiểm tra xem số tiền khách nhập có hợp lệ không"""
        
        # Tui thấy trong file actions.py của bạn đã có sẵn hàm parse_budget_vnd rồi
        # Nên mình sẽ tận dụng nó để quy đổi slot_value (ví dụ "5tr") ra số tự nhiên luôn
        try:
            parsed_budget = parse_budget_vnd(slot_value)
            
            # Trường hợp 1: Hàm parse không hiểu khách nhập gì (trả về None)
            if parsed_budget is None:
                dispatcher.utter_message(
                    text="Mình chưa hiểu số tiền này lắm. Bạn gõ lại số cụ thể (ví dụ: 5 triệu, 3 củ, 2000k...) giúp mình nha! 🥰"
                )
                return {"budget": None} # Trả về None để ép bot hỏi lại
            
            # Trường hợp 2: Tiền âm hoặc bé hơn 100.000 VNĐ
            if parsed_budget < 100000:
                dispatcher.utter_message(
                    text="Hihi, với ngân sách dưới 100 cành thì hơi khó để mình tìm tour hay khách sạn xịn xò cho bạn mất rồi. Bạn ráng 'bơm' thêm chút đỉnh để chuyến đi trọn vẹn hơn nha! 💸✨"
                )
                return {"budget": None} # Trả về None để ép bot hỏi lại
            
            # Trường hợp 3: Hợp lệ (Lớn hơn hoặc bằng 100k)
            # Trả về chính giá trị slot_value ban đầu để Form lưu vào bộ nhớ
            return {"budget": slot_value}
            
        except Exception as e:
            # Phòng hờ lỗi code không lường trước
            print(f"Lỗi ở hàm validate_budget: {e}")
            dispatcher.utter_message(
                text="Híc, con số này làm mình líu lưỡi mất rồi. Bạn nhập lại số tiền dự kiến giúp mình với nhé! 😅"
            )
            return {"budget": None}

class ActionSearchTourInfo(Action):
    def name(self) -> str: 
        return "action_search_tour_info"

    def run(self, dispatcher, tracker, domain):
        latest_entities = tracker.latest_message.get("entities", [])
        
        # Lục tìm xem trong câu khách VỪA NÓI có địa điểm hay loại hình mới không?
        new_dest = next((e.get("value") for e in latest_entities if e.get("entity") == "destination"), None)
        new_cat = next((e.get("value") for e in latest_entities if e.get("entity") == "category"), None)

        # ƯU TIÊN 1: Dùng từ khóa mới (nếu có). ƯU TIÊN 2: Dùng trí nhớ cũ (Slot)
        destination = new_dest or tracker.get_slot("destination")
        category = new_cat or tracker.get_slot("category") or "khách sạn"
        
        print(f"\n[ACTION ACCOMMODATION LOG] ====================")
        print(f"Khách đang tìm: {category} tại {destination}")
        
        if not destination:
            dispatcher.utter_message(text="Bạn muốn mình tìm chỗ ở tại địa điểm nào nhỉ? (Ví dụ: Vũng Tàu, Sapa...)")
            return []

        # 1. Ánh xạ từ khóa tiếng Việt sang cột 'type' trong Database
        acc_type_filter = None
        if category:
            cat_lower = str(category).lower()
            if "resort" in cat_lower or "nghỉ dưỡng" in cat_lower:
                acc_type_filter = "resort"
            elif "homestay" in cat_lower:
                acc_type_filter = "homestay"
            elif "khách sạn" in cat_lower or "hotel" in cat_lower:
                acc_type_filter = "hotel"
            elif "nhà nghỉ" in cat_lower or "motel" in cat_lower:
                acc_type_filter = "motel"
            elif "villa" in cat_lower or "biệt thự" in cat_lower:
                acc_type_filter = "villa"

        # 2. Truy vấn Database bằng SQL (PostgreSQL)
        try:
            conn = get_connection()
            cur = conn.cursor()

            # Viết câu SQL JOIN kinh điển mà chúng ta đã test
            sql = """
                SELECT a.name, a.type, a.latitude, a.longitude
                FROM accommodations a
                JOIN destinations d ON a.destination_id = d.id
                WHERE d.location ILIKE %s
            """
            params = [f"%{destination}%"]

            # Nếu khách đòi chính xác Resort/Homestay thì lọc thêm
            if acc_type_filter:
                sql += " AND a.type = %s"
                params.append(acc_type_filter)

            # Giới hạn 5 kết quả để Bot không chat một tràng dài sọc
            sql += " LIMIT 5;" 

            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

            cur.close()
            conn.close()

            # 3. Xử lý kết quả trả về cho khách
            if not rows:
                msg = f"Mình đã tìm kỹ nhưng hiện tại chưa thấy {category or 'chỗ ở'} nào ở {destination} trong hệ thống. Bạn thử đổi địa điểm khác xem sao nhé!"
                dispatcher.utter_message(text=msg)
                return []

            # Format văn bản Bot trả lời
            type_display = category.title() if category else "Chỗ ở"
            response_text = f"🏨 Dưới đây là một số **{type_display}** tại **{destination}** mình tìm được cho bạn:\n\n"
            
            for row in rows:
                name, type_str, lat, lng = row
                
                # Dịch ngược type từ tiếng Anh trong DB ra tiếng Việt cho thân thiện
                type_vn = {
                    "hotel": "Khách sạn",
                    "resort": "Khu nghỉ dưỡng",
                    "homestay": "Homestay",
                    "motel": "Nhà nghỉ",
                    "villa": "Villa",
                    "other": "Chỗ ở"
                }.get(type_str, "Chỗ ở")

                response_text += f"🔹 **{name}** ({type_vn})\n"
                
                # MLOps xịn xò: Tận dụng luôn tọa độ lat/lng để gen link Google Maps
                if lat and lng:
                    response_text += f"  📍 [Xem Bản đồ](https://www.google.com/maps/search/?api=1&query={lat},{lng})\n"
                response_text += "\n"

            dispatcher.utter_message(text=response_text.strip())

        except Exception as e:
            print(f"[ERROR] Database query failed: {e}")
            dispatcher.utter_message(text="Hệ thống cơ sở dữ liệu đang bảo trì chút xíu, bạn hỏi lại sau nhé! 🛠️")

        print("===========================================\n")
        return [
            SlotSet("destination", destination),
            SlotSet("category", category)
        ]

class ActionSearchTravel(Action):

    def name(self) -> str:
        return "action_search_travel"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> List[Dict[Text, Any]]:
        
        user_message = (tracker.latest_message.get("text") or "").lower()
        latest_entities = tracker.latest_message.get("entities", [])
        predicted_intent = tracker.latest_message.get("intent", {}).get("name")

        print("\n[ACTION SEARCH TRAVEL LOG] " + "="*25)
        print(f"USER NÓI: '{user_message}'")
        print(f"INTENT NHẬN DIỆN: {predicted_intent}")
        
        print("ENTITIES BẮT ĐƯỢC TỪ CÂU NÀY:")
        if not latest_entities:
            print("   -> [Trống]")
        else:
            for e in latest_entities:
                print(f"   -> {e.get('entity')}: '{e.get('value')}' ({e.get('extractor')})")
                
        print("TRÍ NHỚ (SLOTS) ĐANG CÓ:")
        active_slots = {k: v for k, v in tracker.slots.items() if v is not None}
        if not active_slots:
            print("   -> [Trống]")
        else:
            for k, v in active_slots.items():
                print(f"   -> {k}: '{v}'")
        print("="*52 + "\n")

        # 1. TÓM GỌN TẤT CẢ CÁC ENTITY MÀ RASA BẮT ĐƯỢC
        
        # Dùng Dictionary Comprehension để gom toàn bộ các entity vừa bắt được thành 1 cục dễ nhìn
        new_entities = {e.get("entity"): e.get("value") for e in latest_entities}

        # NGUYÊN TẮC: Ưu tiên lấy từ khóa MỚI trong câu khách vừa nói. 
        # Nếu khách không nhắc đến (trả về None), thì tự động lôi Slot CŨ ra xài lại.
        destination_value = new_entities.get("destination") or tracker.get_slot("destination")
        budget_value      = new_entities.get("budget")      or tracker.get_slot("budget")
        season_value      = new_entities.get("season")      or tracker.get_slot("season")
        month_value       = new_entities.get("month")       or tracker.get_slot("month")
        month_from_value  = new_entities.get("month_from")  or tracker.get_slot("month_from")
        month_to_value    = new_entities.get("month_to")    or tracker.get_slot("month_to")
        time_window_value = new_entities.get("time_window") or tracker.get_slot("time_window")
        departure_value   = new_entities.get("departure")   or tracker.get_slot("departure")
        category_value    = new_entities.get("category")    or tracker.get_slot("category")
        duration_value    = new_entities.get("duration")    or tracker.get_slot("duration")


        # 2. CHUẨN HÓA DỮ LIỆU
        month_start, month_end = parse_month_range(
            season_value, month_value, month_from_value, month_to_value, time_window_value
        )
        max_budget = parse_budget_vnd(budget_value)

        # 3. GIA CỐ CÂU QUERY
        enriched_query = user_message
        if destination_value: enriched_query += f" {destination_value}"
        if departure_value: enriched_query += f" từ {departure_value}"
        if category_value: enriched_query += f" {category_value}"

        reset_events = [
            SlotSet("destination", destination_value),
            SlotSet("budget", budget_value),
            SlotSet("category", category_value),
            SlotSet("season", season_value),
            SlotSet("month", month_value),
            SlotSet("time_window", time_window_value),
            SlotSet("departure", departure_value),
            SlotSet("duration", duration_value)
        ]

        try:
            # Truyền câu query đã được bơm thêm thông tin vào hàm search
            results = search_destinations(
                query=enriched_query,
                month_start=month_start,
                month_end=month_end,
                max_budget=max_budget,
                destination=destination_value,
            )
        except Exception as e:
            print(f"Lỗi truy vấn DB: {e}")
            dispatcher.utter_message(text="Hệ thống tìm kiếm tạm thời gián đoạn. Bạn thử lại sau ít phút giúp mình nhé.")
            return []

        if not results:
            dispatcher.utter_message(text="Xin lỗi, tôi chưa tìm thấy địa điểm/tour nào hoàn toàn khớp với yêu cầu của bạn.")
            return reset_events
        if results:
            top_location = results[0].get("location")
            if top_location:
                reset_events.append(SlotSet("destination", top_location))

        # 4. GỌI AI RESPONSE (Để GenAI chém gió lại cho mượt)
        full_context_query = f"Tôi muốn tìm tour đi {destination_value or 'du lịch'}."
        if budget_value:
            full_context_query += f" Ngân sách của tôi là {budget_value}."
        if season_value or month_value:
            full_context_query += f" Tôi dự định đi vào {season_value or ''} {month_value or ''}."

        try:
            ai_response = generate_grounded_ai_response(
                query=full_context_query, 
                results=results,
                month_start=month_start,
                month_end=month_end,
                max_budget=max_budget,
            )
            
            if isinstance(ai_response, dict):
                final_text = ai_response.get("intro", "")
            else:
                final_text = str(ai_response)

            if ai_response is not None and final_text and final_text != "None":
                dispatcher.utter_message(text=final_text.strip())
                return reset_events
            else:
                print("⚠️ Cảnh báo: AI trả về rỗng hoặc None, chuyển sang Fallback DB.")

        except Exception as e:
            print(f"Lỗi GenAI: {e}")

        # 5. FALLBACK: NẾU AI TẠCH, IN RA TEXT THUẦN (như bạn đang thấy)
        response = "Đây là kết quả tìm kiếm trực tiếp từ hệ thống:\n"
        for r in results[:3]:
            response += f"- {r.get('location', 'Không rõ')} | Giá: {r.get('cost', 'Liên hệ')}\n"
            response += f"  {r.get('description', '')[:100]}...\n\n"

        dispatcher.utter_message(text=response)
        return reset_events
