import os
import logging
import json
import re
import requests
from typing import List, Optional

from config.settings import GEMINI_MODEL

logger = logging.getLogger(__name__)


def _normalize(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_json_payload(text: str) -> Optional[dict]:
    cleaned = _strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _match_locations(results: List[dict], locations: List[str]) -> List[dict]:
    normalized_locations = [_normalize(name) for name in locations if str(name).strip()]
    if not normalized_locations:
        return []
    matched = []
    used_indexes = set()
    for candidate_name in normalized_locations:
        for index, item in enumerate(results):
            if index in used_indexes:
                continue
            location = _normalize(item.get("location"))
            if not location:
                continue
            if candidate_name in location or location in candidate_name:
                matched.append(item)
                used_indexes.add(index)
                break
    return matched


def _format_structured_response(
    intro: str, selected_results: List[dict]
) -> Optional[str]:
    if not selected_results:
        return None
    lines = []
    clean_intro = (intro or "").strip()
    if clean_intro:
        lines.append(clean_intro)
    for item in selected_results[:3]:
        lines.append(
            f"- {item.get('location')} | {item.get('cost')} | {item.get('season')}\n"
            f"  {item.get('description')}"
        )
    return "\n".join(lines).strip()


def _format_filters(
    month_start: Optional[int], month_end: Optional[int], max_budget: Optional[int]
) -> str:
    filters = []
    if month_start is not None and month_end is not None:
        filters.append(f"tháng {month_start}-{month_end}")
    if max_budget is not None:
        filters.append(f"ngân sách <= {max_budget:,} VNĐ".replace(",", "."))
    return ", ".join(filters) if filters else "không có ràng buộc cụ thể"


def generate_genz_ai_consultant(
    query: str, intent_name: str, db_context: list = None
) -> Optional[str]:
    """Hàm gọi Gemini để trả lời các câu hỏi phụ (Ăn uống, di chuyển, policy, lịch trình) mang phong cách Gen Z"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Gemini disabled: GEMINI_API_KEY is missing.")
        return (
            "Xin lỗi bạn, trợ lý AI của mình đang ngưng kết nối. Bạn thử hỏi lại nha!"
        )

    context_str = ""
    if db_context and len(db_context) > 0:
        context_str = (
            "\nDỮ LIỆU TỪ HỆ THỐNG CỦA CÔNG TY (Quan trọng):\n"
            + "\n".join(db_context)
            + "\nNẾU CÂU HỎI LIÊN QUAN, HÃY DỰA VÀO DỮ LIỆU NÀY để trả lời. Không khuyên khách đi ra ngoài phạm vi nếu dữ liệu đã cung cấp đủ.\n"
        )
    else:
        context_str = "\n[LƯU Ý: Hiện không tìm thấy dữ liệu nội bộ liên quan. Nếu khách hỏi thông tin cụ thể (ẩm thực, lịch trình, chính sách tour), BẮT BUỘC PHẢI THÔNG BÁO BẠN KHÔNG CÓ THÔNG TIN CỤ THỂ, VÀ GỢI Ý KHÁCH HÀNG TÌM KIẾM ĐỊA ĐIỂM KHÁC. TUYỆT ĐỐI KHÔNG TỰ BỊA RA LỊCH TRÌNH HAY ĐỊA DIỂM TỪ INTERNET.]\n"

    prompt = (
        "Bạn là một 'thổ địa' du lịch mạng phong cách Gen Z, rất sành điệu, "
        "nói chuyện siêu cuốn, thân thiện dễ thương. Xưng hô 'mình' - 'bạn'. "
        "Thỉnh thoảng có thể dùng từ lóng nhẹ nhàng (ví dụ: cực dính, nhức nách, hạt dẻ, cháy máy, chữa lành...) "
        "nhưng TUYỆT ĐỐI không được trẻ trâu hay quá đà, giữ thái độ lịch sự.\n\n"
        f"Ngữ cảnh câu hỏi thuộc dạng: {intent_name}\n"
        f"Câu hỏi của khách: {query}\n"
        f"{context_str}\n\n"
        "Nhiệm vụ: Hãy tư vấn, giải đáp ngắn gọn, súc tích (dưới 150 chữ). Ưu tiên RÚT TRÍCH TỪ PHẦN DỮ LIỆU TỪ HỆ THỐNG BÊN TRÊN. KHÔNG bịa đặt dữ liệu (hallucination)."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "responseMimeType": "text/plain"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return (
            text.strip()
            if text
            else "Ca này khó ta, bạn cho mình xin thêm tí thông tin với nha!"
        )
    except Exception as e:
        logger.error(f"GenZ AI Consultant error: {e}")
        return "Bảo trì tí xíu ạ, LLM đang nghẽn xíu, bạn chờ xíu nha! 🛠️"


def generate_grounded_ai_response(
    query: str,
    results: List[dict],
    month_start: Optional[int] = None,
    month_end: Optional[int] = None,
    max_budget: Optional[int] = None,
) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Gemini disabled: GEMINI_API_KEY is missing.")
        return None

    if not results:
        return None

    context_lines = []
    for index, item in enumerate(results[:5], start=1):
        context_lines.append(
            f"{index}. {item.get('location')} | {item.get('cost')} | {item.get('season')} | {item.get('description')}"
        )

    prompt = (
        "Bạn là một 'Travel Blogger' du lịch phong cách Gen Z, rất sành điệu, "
        "nói chuyện siêu cuốn, thân thiện nhưng không bị 'trẻ trâu', xưng hô 'mình' - 'bạn'. "
        "Thỉnh thoảng dùng vài từ lóng nhẹ nhàng (như: cực dính, nhức nách, hạt dẻ, sống ảo cháy máy, healing...) "
        "để lời tư vấn thêm mượt mà.\n\n"
        "Hãy dựa vào danh sách địa điểm dưới đây để tư vấn cho khách. TUYỆT ĐỐI không bịa thêm thông tin.\n\n"
        f"Câu hỏi người dùng: {query}\n"
        f"Bộ lọc: {_format_filters(month_start, month_end, max_budget)}\n\n"
        "Danh sách điểm đến phù hợp từ Database:\n"
        + "\n".join(context_lines)
        + (
            "\n\nNhiệm vụ:\n"
            "1) Chọn tối đa 3 địa điểm phù hợp nhất từ danh sách trên.\n"
            "2) Trả về kết quả dưới định dạng JSON.\n"
            "3) Trường 'intro' hãy viết 1-2 câu tư vấn thật natural, cool ngầu.\n"
            "4) Định dạng bắt buộc: "
            '{"intro": "<Câu bình luận Gen Z>", "locations": ["<tên địa điểm 1>", "<tên địa điểm 2>"]}'
        )
    )

    # GỌI TRỰC TIẾP REST API BẰNG REQUESTS (Bypass SDK)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Bắt lỗi nếu HTTP rớt (400, 401, 500...)
        data = response.json()

        # Bóc tách text từ chuỗi JSON phản hồi của Google
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        print(f"\n[GEN-AI RESPONSE RAW]:\n{text}\n")

        if text and text.strip():
            json_payload = _extract_json_payload(text)
            if not json_payload:
                logger.warning("Gemini returned non-JSON content.")
                return None

            intro = str(json_payload.get("intro", "")).strip()
            locations = json_payload.get("locations", [])
            if not isinstance(locations, list):
                logger.warning("Gemini JSON payload has invalid 'locations' field.")
                return None

            matched_results = _match_locations(
                results, [str(name) for name in locations]
            )
            if not matched_results:
                logger.warning("Gemini locations do not match retrieval results.")
                return None

            structured_response = _format_structured_response(intro, matched_results)
            if structured_response:
                return structured_response

        logger.warning("Gemini returned empty text response.")
    except Exception as exc:
        logger.warning(f"Gemini REST API failed: {exc}")
        return None

    return None


def _match_tour_names(results: List[dict], tour_names: List[str]) -> List[dict]:
    normalized_names = [_normalize(name) for name in tour_names if str(name).strip()]
    if not normalized_names:
        return []
    matched = []
    used_indexes = set()
    for candidate_name in normalized_names:
        for index, item in enumerate(results):
            if index in used_indexes:
                continue
            item_name = _normalize(item.get("tour_name"))
            if not item_name:
                continue
            if candidate_name in item_name or item_name in candidate_name:
                matched.append(item)
                used_indexes.add(index)
                break
    return matched


def _format_tour_response(intro: str, selected_results: List[dict]) -> Optional[str]:
    clean_intro = (intro or "").strip()
    if not selected_results:
        return clean_intro if clean_intro else None
    lines = []
    if clean_intro:
        lines.append(clean_intro)
    lines.append("")
    for item in selected_results[:3]:
        tour_name = item.get("tour_name", "Tour")
        price = item.get("price")
        price_str = f"{price:,} VND" if price else "Liên hệ"
        duration = item.get("duration_text") or f"{item.get('days', '?')} ngày {item.get('nights', '?')} đêm"
        dests = ", ".join(item.get("destinations", [])) if item.get("destinations") else ""
        dep = item.get("departure") or ""
        transport = item.get("transportation") or ""

        lines.append(f"🔹 **{tour_name}**")
        lines.append(f"  💰 {price_str} | ⏱ {duration}")
        if dests:
            lines.append(f"  📍 {dests}")
        if dep:
            lines.append(f"  🏁 {dep}")
        if transport:
            lines.append(f"  🚌 {transport}")
        lines.append("")
    return "\n".join(lines).strip()


def generate_tour_ai_response(
    query: str,
    results: List[dict],
) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Gemini disabled: GEMINI_API_KEY is missing.")
        return None

    if not results:
        return None

    context_lines = []
    for index, item in enumerate(results[:5], start=1):
        tour_name = item.get("tour_name", "Không rõ")
        price = item.get("price")
        price_str = f"{price:,} VND" if price else "Liên hệ"
        duration = item.get("duration_text") or f"{item.get('days', '?')} ngày {item.get('nights', '?')} đêm"
        dests = ", ".join(item.get("destinations", [])) if item.get("destinations") else "Không rõ"
        dep = item.get("departure") or "Không rõ"
        transport = item.get("transportation") or "Không rõ"

        rag_kb = item.get("rag_knowledge_base") or {}
        itinerary = rag_kb.get("itinerary", []) if isinstance(rag_kb, dict) else []
        itinerary_text = ""
        for day in itinerary:
            day_title = day.get("title", "")
            day_desc = day.get("description", "")
            itinerary_text += f"      - {day_title}: {day_desc[:200]}\n"

        context_lines.append(
            f"{index}. Tour: {tour_name}\n"
            f"   Giá: {price_str}\n"
            f"   Thời gian: {duration}\n"
            f"   Điểm đến: {dests}\n"
            f"   Khởi hành: {dep}\n"
            f"   Phương tiện: {transport}\n"
            f"   Lịch trình chi tiết:\n{itinerary_text}"
        )

    context_str = "\n".join(context_lines)

    prompt = (
        "Bạn là một 'Travel Blogger' du lịch phong cách Gen Z, rất sành điệu, "
        "nói chuyện siêu cuốn, thân thiện nhưng không bị 'trẻ trâu', xưng hô 'mình' - 'bạn'. "
        "Thỉnh thoảng dùng vài từ lóng nhẹ nhàng (như: cực dính, nhức nách, hạt dẻ, sống ảo cháy máy, healing...) "
        "để lời tư vấn thêm mượt mà.\n\n"
        "Hãy dựa vào danh sách tour dưới đây (BAO GỒM LỊCH TRÌNH CHI TIẾT) để trả lời câu hỏi của khách. "
        "TUYỆT ĐỐI không bịa thêm thông tin không có trong dữ liệu được cung cấp. "
        "Nếu câu hỏi của khách yêu cầu thông tin không có trong dữ liệu, hãy nói 'Mình không có thông tin này trong hệ thống'.\n\n"
        f"Câu hỏi người dùng: {query}\n\n"
        "Danh sách tour từ Database:\n"
        f"{context_str}\n\n"
        "Nhiệm vụ:\n"
        "1) Chọn tối đa 3 tour phù hợp nhất từ danh sách trên.\n"
        "2) Trả về kết quả dưới định dạng JSON.\n"
        "3) Trường 'intro' hãy viết 1-2 câu tư vấn thật natural, trả lời đúng trọng tâm câu hỏi của khách, "
        "dùng thông tin cụ thể từ dữ liệu tour (giá, lịch trình, điểm đến...) và đề xuất tour phù hợp.\n"
        "4) Định dạng bắt buộc: "
        '{"intro": "<Câu tư vấn ngắn gọn>", "tours": ["<tên tour 1>", "<tên tour 2>"]}'
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        print(f"\n[GEN-AI TOUR RESPONSE RAW]:\n{text}\n")

        if text and text.strip():
            json_payload = _extract_json_payload(text)
            if not json_payload:
                logger.warning("Gemini returned non-JSON content for tour.")
                return None

            intro = str(json_payload.get("intro", "")).strip()
            tour_names = json_payload.get("tours", [])
            if not isinstance(tour_names, list):
                logger.warning("Gemini JSON payload has invalid 'tours' field.")
                return None

            matched_results = _match_tour_names(
                results, [str(name) for name in tour_names]
            )
            if not matched_results:
                if not intro:
                    logger.warning("Gemini returned empty intro and no matching tours.")
                    return None
                # Gemini correctly identified no exact matches (e.g. wrong duration).
                # Keep the intro but append alternative suggestions from full results.
                response = intro.strip()
                if results:
                    alt_lines = ["\n\nTuy nhiên, mình có một số gợi ý khác cho bạn nè:"]
                    for item in results[:3]:
                        tour_name = item.get("tour_name", "Tour")
                        price = item.get("price")
                        price_str = f"{price:,} VND" if price else "Liên hệ"
                        duration = item.get("duration_text") or f"{item.get('days', '?')} ngày {item.get('nights', '?')} đêm"
                        dep = item.get("departure") or ""
                        alt_lines.append(f"\n🔹 **{tour_name}**  💰 {price_str}  ⏱ {duration}" + (f"  🏁 {dep}" if dep else ""))
                    response += "".join(alt_lines)
                return response

            structured_response = _format_tour_response(intro, matched_results)
            if structured_response:
                return structured_response

        logger.warning("Gemini returned empty text response for tour.")
    except Exception as exc:
        logger.warning(f"Gemini tour REST API failed: {exc}")
        return None

    return None
