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

from services.search_service import search_destinations, search_tours
from services.ai_response_service import (
    generate_grounded_ai_response,
    generate_genz_ai_consultant,
    generate_tour_ai_response,
)

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from database.db_connection import get_connection

ALL_SLOTS = [
    "season", "month", "month_from", "month_to", "budget",
    "time_window", "destination", "category", "departure",
    "time", "party_size", "tour_name", "duration",
]


def _reset_on_command(dispatcher, tracker):
    """Nếu user gõ 'reset slot' thì xoá toàn bộ slots và return events."""
    user_message = (tracker.latest_message.get("text") or "").strip().lower()
    if user_message == "reset slot":
        dispatcher.utter_message(
            text="🔄 Đã reset toàn bộ dữ liệu phiên làm việc! Bạn có thể hỏi lại nhé."
        )
        return [SlotSet(slot, None) for slot in ALL_SLOTS]
    return None


SKIP_CATEGORY_WORDS = {"sao", "thêm", "đi", "nhé", "nha", "ní", "dợ", "bot", "vậy", "thì"}
LOCATION_NOISE_WORDS = {"bằng", "ra", "qua", "vào", "từ"}


def _first_valid_category(entities):
    for e in entities:
        if e.get("entity") == "category":
            val = str(e.get("value", "")).strip().lower()
            if val not in SKIP_CATEGORY_WORDS:
                return val
    return None


def _first_valid_location(entities, role="destination"):
    """Get first location entity by role, excluding noise words."""
    valid_roles = ("destination", None) if role == "destination" else ("departure",)
    for e in entities:
        if e.get("entity") == "location" and e.get("role") in valid_roles:
            val = str(e.get("value", "")).strip().lower()
            if val not in LOCATION_NOISE_WORDS:
                return e.get("value")
    return None


CATEGORY_KEYWORDS = {
    "resort": "resort", "nghỉ dưỡng": "resort", "khu nghỉ dưỡng": "resort",
    "homestay": "homestay",
    "khách sạn": "hotel", "hotel": "hotel",
    "nhà nghỉ": "motel", "motel": "motel",
    "villa": "villa", "biệt thự": "villa",
    "chỗ ở": "hotel", "chỗ nghỉ": "hotel", "phòng": "hotel",
}


def _match_category_from_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    for keyword, value in CATEGORY_KEYWORDS.items():
        if keyword in text_lower:
            return value
    return None


def _match_destination_from_text(text: str) -> Optional[str]:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT location FROM destinations")
        known_locations = [row[0].lower() for row in cur.fetchall()]
        cur.close()
        conn.close()
        text_lower = text.lower()
        for loc in sorted(known_locations, key=len, reverse=True):
            if loc in text_lower:
                return loc
    except Exception as e:
        print(f"[MATCH DESTINATION] DB error: {e}")
    return None


def _log_action(
    name: str,
    user_message: str,
    intent: str,
    entities: list,
    slots: dict,
):
    print(f"\n[{name} LOG] " + "=" * 30, flush=True)
    print(f"USER NÓI : '{user_message}'")
    print(f"INTENT   : {intent}")
    print()
    print("ENTITIES BẮT ĐƯỢC:")
    if not entities:
        print("   -> [Trống]")
    else:
        for e in entities:
            print(f"   -> {e.get('entity')}: '{e.get('value')}'  [{e.get('extractor')}]")
    active = {k: v for k, v in slots.items() if v is not None}
    print()
    print("SLOTS ĐANG GIỮ:")
    if not active:
        print("   -> [Trống]")
    else:
        for k, v in active.items():
            print(f"   -> {k}: '{v}'")
    print("=" * 52 + "\n")


SEASON_TO_MONTHS = {
    "mùa xuân": (1, 3),
    "mua xuan": (1, 3),
    "mùa hè": (4, 6),
    "mua he": (4, 6),
    "mùa thu": (7, 9),
    "mua thu": (7, 9),
    "mùa đông": (10, 12),
    "mua dong": (10, 12),
}


def parse_budget_vnd(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None

    text = str(value).lower().strip()

    unit_match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(triệu|trieu|tr|củ|k|nghìn|ngàn|cành|lít|vnd|đ|dong)?(?:\s*(rưỡi|ruoi))?",
        text,
    )

    if unit_match:
        val = float(unit_match.group(1).replace(",", "."))
        unit = unit_match.group(2)
        has_half = unit_match.group(3) is not None
        
        if has_half:
            val += 0.5

        if unit in {"triệu", "trieu", "tr", "củ"}:
            return int(val * 1_000_000)
        if unit in {"lít"}:
            return int(val * 100_000)
        if unit in {"k", "nghìn", "ngàn", "cành"}:
            return int(val * 1_000)
        if unit in {"vnd", "đ", "dong"}:
            return int(val)
        if val < 1000:
            return int(val * 1_000_000)
        return int(val)

    return None

    text = str(value).lower().strip()

    unit_match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*(triệu|trieu|tr|củ|k|nghìn|ngàn|cành|lít|vnd|đ|dong)?",
        text,
    )

    if unit_match:
        value = float(unit_match.group(1).replace(",", "."))
        unit = unit_match.group(2)

        if unit in {"triệu", "trieu", "tr", "củ"}:
            return int(value * 1_000_000)
        if unit in {"lít"}:
            return int(value * 100_000)
        if unit in {"k", "nghìn", "ngàn", "cành"}:
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
        return "validate_travel_form"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text", "")
        intent = tracker.latest_message.get("intent", {}).get("name")
        entities = tracker.latest_message.get("entities", [])

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset + [SlotSet("requested_slot", None)]

        _log_action("FORM VALIDATION", user_message, intent, entities, tracker.slots)

        skip_intents = {"out_of_scope", "bot_challenge", "goodbye", "search_food_dining",
                        "ask_transportation", "ask_policy_booking", "ask_itinerary",
                        "search_activity", "search_accommodation", "ask_location_feature"}
        if intent in skip_intents:
            dispatcher.utter_message(
                text="Có vẻ bạn muốn hỏi chuyện khác trước. Khi nào cần tìm tour, cứ gọi mình nha! 😊"
            )
            return [SlotSet("destination", None), SlotSet("budget", None),
                    SlotSet("requested_slot", None)]

        return await super().run(dispatcher, tracker, domain)

    def validate_destination(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if slot_value:
            return {"destination": slot_value}

        user_text = tracker.latest_message.get("text", "")
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT location FROM destinations")
            known_locations = [row[0].lower() for row in cur.fetchall()]
            cur.close()
            conn.close()

            user_lower = user_text.lower()
            for loc in sorted(known_locations, key=len, reverse=True):
                if loc in user_lower:
                    return {"destination": loc}
        except Exception as e:
            print(f"[VALIDATE DESTINATION] DB error: {e}")

        return {"destination": None}

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
                return {"budget": None}  # Trả về None để ép bot hỏi lại

            # Trường hợp 2: Tiền âm hoặc bé hơn 100.000 VNĐ
            if parsed_budget < 100000:
                dispatcher.utter_message(
                    text="Hihi, với ngân sách dưới 100 cành thì hơi khó để mình tìm tour hay khách sạn xịn cho bạn mất rồi. Ví đang mỏng đúng hông ta? 'Bơm' thêm chút đỉnh để đi chơi cho cháy nhé! 💸✨"
                )
                return {"budget": None}  # Trả về None để ép bot hỏi lại

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
        user_message = (tracker.latest_message.get("text") or "").lower()
        latest_entities = tracker.latest_message.get("entities", [])

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset

        _log_action("ACTION SEARCH TOUR INFO", user_message, tracker.latest_message.get("intent", {}).get("name"), latest_entities, tracker.slots)

        new_dest = _first_valid_location(latest_entities, role="destination")
        new_cat = _first_valid_category(latest_entities)

        # ƯU TIÊN 1: Dùng từ khóa mới (nếu có). ƯU TIÊN 2: Dùng trí nhớ cũ (Slot)
        destination = new_dest or tracker.get_slot("destination")
        # Fallback: text-match destination từ user message
        if not destination:
            destination = _match_destination_from_text(user_message)
        # Nếu slot cũ chứa từ noise thì bỏ qua, dùng fallback
        old_cat = tracker.get_slot("category")
        if old_cat and str(old_cat).strip().lower() in SKIP_CATEGORY_WORDS:
            old_cat = None
        # Thêm text-matching fallback cho category (nếu NLU không extract được)
        text_cat = _match_category_from_text(user_message) if not (new_cat or old_cat) else None
        category = new_cat or old_cat or text_cat or "khách sạn"

        print(f"→ Tìm: {category} tại {destination}")
        print("=" * 52 + "\n")

        if not destination:
            dispatcher.utter_message(
                text="Bạn muốn mình tìm chỗ ở tại địa điểm nào nhỉ? (Ví dụ: Vũng Tàu, Sapa...)"
            )
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
                    "other": "Chỗ ở",
                }.get(type_str, "Chỗ ở")

                response_text += f"🔹 **{name}** ({type_vn})\n"

                # MLOps xịn xò: Tận dụng luôn tọa độ lat/lng để gen link Google Maps
                if lat and lng:
                    response_text += f"  📍 [Xem Bản đồ](https://www.google.com/maps/search/?api=1&query={lat},{lng})\n"
                response_text += "\n"

            dispatcher.utter_message(text=response_text.strip())
            dispatcher.utter_message(response="utter_suggest_more")

        except Exception as e:
            print(f"[ERROR] Database query failed: {e}")
            dispatcher.utter_message(
                text="Hệ thống cơ sở dữ liệu đang bảo trì chút xíu, bạn hỏi lại sau nhé! 🛠️"
            )

        print("===========================================\n")
        return [SlotSet("destination", destination), SlotSet("category", category)]


class ActionSearchTravel(Action):

    def name(self) -> str:
        return "action_search_travel"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:

        user_message = (tracker.latest_message.get("text") or "").lower()
        latest_entities = tracker.latest_message.get("entities", [])
        predicted_intent = tracker.latest_message.get("intent", {}).get("name")

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset

        _log_action("ACTION SEARCH TRAVEL", user_message, predicted_intent, latest_entities, tracker.slots)

        # GUARD: Intent out_of_scope → form bị hủy, không search
        if predicted_intent in {"out_of_scope", "bot_challenge"}:
            print(f"⚠️ Bỏ qua action_search_travel vì intent='{predicted_intent}' (form bị hủy)")
            return []

        # GUARD: "hỏi thêm" không có destination → không search
        if predicted_intent == "inform" and "hỏi thêm" in user_message:
            dispatcher.utter_message(
                text="Bạn muốn hỏi thêm gì về điểm đến hay tour nào không? Mình sẵn sàng tư vấn! 😊"
            )
            return []

        # 1. TÓM GỌN TẤT CẢ CÁC ENTITY MÀ RASA BẮT ĐƯỢC

        # Dùng Dictionary Comprehension để gom toàn bộ các entity vừa bắt được thành 1 cục dễ nhìn
        new_entities = {e.get("entity"): e.get("value") for e in latest_entities}

        # Role-aware location extraction
        destination_value = (
            _first_valid_location(latest_entities, role="destination")
            or tracker.get_slot("destination")
        )
        departure_value = (
            _first_valid_location(latest_entities, role="departure")
            or tracker.get_slot("departure")
        )

        budget_value = new_entities.get("budget") or tracker.get_slot("budget")
        season_value = new_entities.get("season") or tracker.get_slot("season")
        month_value = new_entities.get("month") or tracker.get_slot("month")
        month_from_value = new_entities.get("month_from") or tracker.get_slot(
            "month_from"
        )
        month_to_value = new_entities.get("month_to") or tracker.get_slot("month_to")
        time_window_value = new_entities.get("time_window") or tracker.get_slot(
            "time_window"
        )
        category_value = (
            _first_valid_category(latest_entities)
            or tracker.get_slot("category")
        )
        duration_value = new_entities.get("duration") or tracker.get_slot("duration")

        # 2. CHUẨN HÓA DỮ LIỆU
        month_start, month_end = parse_month_range(
            season_value,
            month_value,
            month_from_value,
            month_to_value,
            time_window_value,
        )
        max_budget = parse_budget_vnd(budget_value)

        # 3. GIA CỐ CÂU QUERY
        enriched_query = user_message
        if destination_value:
            enriched_query += f" {destination_value}"
        if departure_value:
            enriched_query += f" từ {departure_value}"
        if category_value:
            enriched_query += f" {category_value}"

        reset_events = [
            SlotSet("destination", destination_value),
            SlotSet("budget", budget_value),
            SlotSet("category", category_value),
            SlotSet("season", season_value),
            SlotSet("month", month_value),
            SlotSet("time_window", time_window_value),
            SlotSet("departure", departure_value),
            SlotSet("duration", duration_value),
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
            dispatcher.utter_message(
                text="Hệ thống tìm kiếm tạm thời gián đoạn. Bạn thử lại sau ít phút giúp mình nhé."
            )
            return []

        if not results:
            suggestions = []
            if destination_value:
                suggestions.append("thử một địa điểm khác")
            if max_budget is not None:
                suggestions.append("tăng ngân sách lên một chút")
            if month_start is not None or month_end is not None:
                suggestions.append("đổi thời gian đi")
            if not suggestions:
                suggestions.append("cho mình thêm thông tin để gợi ý chuẩn hơn")

            msg = "Mình đã lục tung database nhưng chưa tìm thấy địa điểm nào khớp với yêu cầu của bạn."
            if suggestions:
                msg += f" Bạn thử {' hoặc '.join(suggestions)} nhé!"
            dispatcher.utter_message(text=msg)
            return reset_events

        # Chỉ set destination từ top result khi có search intent thực sự (tránh slot bleed)
        SEARCH_INTENTS = {"search_travel", "search_destination", "search_price"}
        if predicted_intent in SEARCH_INTENTS and results:
            top_location = results[0].get("location")
            if top_location:
                reset_events.append(SlotSet("destination", top_location))

        # 4. GỌI AI RESPONSE (Để GenAI chém gió lại cho mượt)
        full_context_query = f"Tôi muốn tìm tour đi {destination_value or 'du lịch'}."
        if budget_value:
            full_context_query += f" Ngân sách của tôi là {budget_value}."
        if season_value or month_value:
            full_context_query += (
                f" Tôi dự định đi vào {season_value or ''} {month_value or ''}."
            )

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
                dispatcher.utter_message(response="utter_suggest_more")
                return reset_events
            else:
                print("⚠️ Cảnh báo: AI trả về rỗng hoặc None, chuyển sang Fallback DB.")

        except Exception as e:
            print(f"Lỗi GenAI: {e}")

        # 5. FALLBACK: NẾU AI TẠCH, IN RA TEXT THUẦN (như bạn đang thấy)
        response = "🔍 **Mình tìm thấy một số điểm đến phù hợp:**\n\n"
        for r in results[:3]:
            response += (
                f"📍 **{r.get('location', 'Không rõ')}**\n"
            )
            response += f"  💰 Giá: {r.get('cost', 'Liên hệ')}\n"
            desc = r.get('description', '')[:200]
            if desc:
                response += f"  {desc}\n"
            response += "\n"

        dispatcher.utter_message(text=response)
        dispatcher.utter_message(response="utter_suggest_more")
        return reset_events


class ActionSearchTour(Action):

    def name(self) -> str:
        return "action_search_tour"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        user_message = (tracker.latest_message.get("text") or "").lower()
        latest_entities = tracker.latest_message.get("entities", [])
        predicted_intent = tracker.latest_message.get("intent", {}).get("name")

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset

        _log_action("ACTION SEARCH TOUR", user_message, predicted_intent, latest_entities, tracker.slots)

        new_entities = {e.get("entity"): e.get("value") for e in latest_entities}

        destination = (
            _first_valid_location(latest_entities, role="destination")
            or tracker.get_slot("destination")
        )
        # Fallback: text-match destination từ user message
        if not destination:
            destination = _match_destination_from_text(user_message)
        departure_value = (
            _first_valid_location(latest_entities, role="departure")
            or tracker.get_slot("departure")
        )
        budget_value = new_entities.get("budget") or tracker.get_slot("budget")
        category_value = new_entities.get("category") or tracker.get_slot("category")
        duration_value = new_entities.get("duration") or tracker.get_slot("duration")

        max_budget = parse_budget_vnd(budget_value)

        # Parse duration → số ngày
        duration_days = None
        if duration_value:
            match = re.search(r"(\d+)\s*(?:ngày|ngay)", str(duration_value))
            if match:
                duration_days = int(match.group(1))

        print(f"→ PARSED BUDGET: {max_budget} VND")
        print(f"→ DESTINATION: {destination} | DEPARTURE: {departure_value} | CATEGORY: {category_value} | DURATION: {duration_days} ngày")
        print("=" * 52 + "\n")

        reset_events = [
            SlotSet("destination", destination),
            SlotSet("budget", budget_value),
            SlotSet("departure", departure_value),
            SlotSet("category", category_value),
            SlotSet("duration", duration_value),
        ]

        try:
            print("[QUERY DB] Searching tours...")
            results = search_tours(
                query=user_message,
                max_budget=max_budget,
                destination=destination,
                departure=departure_value,
                duration_days=duration_days,
            )
            print(f"[QUERY DB] {len(results)} tour(s) found.")
        except Exception as e:
            print(f"[QUERY DB] ERROR: {e}")
            dispatcher.utter_message(
                text="Hệ thống tìm tour đang bảo trì tí xíu, bạn thử lại sau nha! 🛠️"
            )
            return reset_events

        if not results:
            print("[RESULT] No tours match the query — sending fallback message.")
            dispatcher.utter_message(
                text="Mình đã tìm kỹ nhưng chưa thấy tour nào phù hợp. Bạn thử đổi địa điểm hoặc ngân sách khác xem sao nhé! 😊"
            )
            return reset_events

        print("[GEN-AI] Calling generate_tour_ai_response...")
        try:
            ai_response = generate_tour_ai_response(
                query=user_message,
                results=results,
            )

            if ai_response:
                print(f"[GEN-AI] OK — response has {len(ai_response.split(chr(10)))} line(s).")
                dispatcher.utter_message(text=ai_response.strip())
                dispatcher.utter_message(response="utter_suggest_more")
                return reset_events
            else:
                print("[GEN-AI] Empty response — falling back to DB text.")

        except Exception as e:
            print(f"[GEN-AI] ERROR: {e}")

        print("[FALLBACK] Building plain-text tour listing...")
        response_parts = ["🎉 Mình tìm thấy một số tour phù hợp cho bạn:\n"]
        for r in results[:5]:
            tour_name = r.get("tour_name", "Tour")
            price = r.get("price")
            price_str = f"{price:,} VND" if price else "Liên hệ"
            duration = r.get("duration_text") or f"{r.get('days', '?')} ngày {r.get('nights', '?')} đêm"
            dests = ", ".join(r.get("destinations", [])) if r.get("destinations") else ""
            dep = r.get("departure") or ""
            transport = r.get("transportation") or ""

            response_parts.append(f"🔹 **{tour_name}**")
            response_parts.append(f"  💰 {price_str} | ⏱ {duration}")
            if dests:
                response_parts.append(f"  📍 {dests}")
            if dep:
                response_parts.append(f"  🏁 Khởi hành: {dep}")
            if transport:
                response_parts.append(f"  🚌 Di chuyển: {transport}")
            response_parts.append("")

        response_text = "\n".join(response_parts).strip()
        print(f"[FALLBACK] Sent {len(results)} tour(s) as plain text.")
        dispatcher.utter_message(text=response_text)
        dispatcher.utter_message(response="utter_suggest_more")
        return reset_events


class ActionSearchActivity(Action):
    def name(self) -> str:
        return "action_search_activity"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text", "")
        intent_name = tracker.latest_message.get("intent", {}).get("name", "unknown")
        latest_entities = tracker.latest_message.get("entities", [])

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset

        _log_action("ACTION SEARCH ACTIVITY", user_message, intent_name, latest_entities, tracker.slots)

        destination = (
            _first_valid_location(latest_entities, role="destination")
            or tracker.get_slot("destination")
        )
        if not destination:
            destination = _match_destination_from_text(user_message)

        results = []
        try:
            results = search_destinations(query=user_message, destination=destination)
        except Exception as e:
            print(f"[DB LOG] Lỗi truy xuất DB: {e}")

        if results:
            query_words = set(re.findall(r"[\wÀ-ỹ]{2,}", user_message.lower()))
            query_words -= {"đi", "ở", "đâu", "nào", "có", "không", "và", "là", "cho",
                            "tôi", "mình", "bạn", "bot", "nha", "nhé", "với", "hay",
                            "thì", "mà", "cứ", "được", "ra", "lắm", "quá"}
            activity_results = []
            for r in results:
                activities = (r.get("activities", "") or "").lower()
                if query_words and any(w in activities for w in query_words):
                    activity_results.append(r)
                elif not query_words and destination:
                    activity_results.append(r)

            if activity_results:
                response = "🎯 **Mình tìm thấy các địa điểm có hoạt động này:**\n\n"
                for r in activity_results[:3]:
                    response += f"📍 **{r.get('location')}**\n"
                    acts = r.get("activities", "")
                    if acts:
                        response += f"  🎯 {acts[:200]}\n"
                    desc = r.get("description", "")
                    if desc:
                        response += f"  {desc[:200]}\n"
                    response += "\n"
                dispatcher.utter_message(text=response.strip())
                dispatcher.utter_message(response="utter_suggest_more")
                print(f"→ Tìm thấy {len(activity_results)} địa điểm có hoạt động phù hợp")
                return []

        print("→ Không tìm thấy kết quả DB, fallback Gemini...")
        try:
            response = generate_genz_ai_consultant(user_message, "search_activity", [])
            dispatcher.utter_message(text=response)
            dispatcher.utter_message(response="utter_suggest_more")
        except Exception as e:
            print(f"[ERROR] AI Consultant failed: {e}")
            dispatcher.utter_message(
                text="Xin lỗi, mình chưa tìm thấy hoạt động phù hợp trong hệ thống."
            )

        print("====================================================\n")
        return [SlotSet("destination", destination)]


class ActionAIConsultant(Action):
    def name(self) -> Text:
        return "action_ai_consultant"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "")
        intent_name = tracker.latest_message.get("intent", {}).get("name", "unknown")
        latest_entities = tracker.latest_message.get("entities", [])

        reset = _reset_on_command(dispatcher, tracker)
        if reset:
            return reset

        _log_action("ACTION AI CONSULTANT", user_message, intent_name, latest_entities, tracker.slots)

        new_dest = _first_valid_location(latest_entities, role="destination")
        destination = new_dest or tracker.get_slot("destination")
        # Fallback: text-match destination từ user message nếu NLU không extract được
        if not destination:
            destination = _match_destination_from_text(user_message)

        db_context = []
        if destination:
            try:
                results = search_destinations(destination=destination)
                for r in results[:2]:
                    db_context.append(
                        f"Điểm đến: {r.get('location')} | Hoạt động giải trí: {r.get('activities')} | Chi tiết: {r.get('description')}"
                    )
            except Exception as e:
                print(f"[DB LOG] Lỗi truy xuất DB: {e}")

        print(f"→ DB context: {len(db_context)} bản ghi | Gọi Gemini...")
        print("=" * 52 + "\n")

        try:
            response = generate_genz_ai_consultant(
                user_message, intent_name, db_context
            )
            dispatcher.utter_message(text=response)
            dispatcher.utter_message(response="utter_suggest_more")
        except Exception as e:
            print(f"[ERROR] AI Consultant failed: {e}")
            dispatcher.utter_message(
                text="Trợ lý hơi lag xíu, bạn chờ mình ngâm cứu thêm xíu nha! 😅"
            )

        print("====================================================\n")
        return [SlotSet("destination", destination)]


class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(
            text="🔄 Đã reset toàn bộ dữ liệu phiên làm việc! Bạn có thể hỏi lại nhé."
        )
        return [SlotSet(slot, None) for slot in ALL_SLOTS] + [
            SlotSet("requested_slot", None)
        ]