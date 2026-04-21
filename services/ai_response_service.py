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

def _format_structured_response(intro: str, selected_results: List[dict]) -> Optional[str]:
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

def _format_filters(month_start: Optional[int], month_end: Optional[int], max_budget: Optional[int]) -> str:
    filters = []
    if month_start is not None and month_end is not None:
        filters.append(f"tháng {month_start}-{month_end}")
    if max_budget is not None:
        filters.append(f"ngân sách <= {max_budget:,} VNĐ".replace(",", "."))
    return ", ".join(filters) if filters else "không có ràng buộc cụ thể"

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
        "Bạn là một trợ lý du lịch nhiệt tình, thân thiện và duyên dáng. "
        "Hãy dựa vào danh sách địa điểm dưới đây để tư vấn cho khách. TUYỆT ĐỐI không bịa thêm thông tin.\n\n"
        f"Câu hỏi người dùng: {query}\n"
        f"Bộ lọc: {_format_filters(month_start, month_end, max_budget)}\n\n"
        "Danh sách điểm đến phù hợp từ Database:\n"
        + "\n".join(context_lines)
        + (
            "\n\nNhiệm vụ:\n"
            "1) Chọn tối đa 3 địa điểm phù hợp nhất từ danh sách trên.\n"
            "2) Trả về kết quả dưới định dạng JSON.\n"
            "3) Trường 'intro' hãy viết 1-2 câu tư vấn thật tự nhiên, xưng 'mình' và 'bạn', thể hiện sự hào hứng.\n"
            "4) Định dạng bắt buộc: "
            '{"intro": "<Câu giới thiệu tự nhiên>", "locations": ["<tên địa điểm 1>", "<tên địa điểm 2>"]}'
        )
    )

    # GỌI TRỰC TIẾP REST API BẰNG REQUESTS (Bypass SDK)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Bắt lỗi nếu HTTP rớt (400, 401, 500...)
        data = response.json()
        
        # Bóc tách text từ chuỗi JSON phản hồi của Google
        text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
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

            matched_results = _match_locations(results, [str(name) for name in locations])
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