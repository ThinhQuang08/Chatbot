import os
import logging
import json
import re
import requests
from typing import List, Optional

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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
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
