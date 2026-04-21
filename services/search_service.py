# services/search_service.py
import re
from typing import List, Optional

from database.db_connection import get_connection
from services.semantic_service import (
    build_destination_candidate_id,
    build_destination_semantic_text,
    semantic_scores,
)


STOPWORDS = {
    "toi",
    "tôi",
    "muon",
    "muốn",
    "di",
    "đi",
    "du",
    "lich",
    "lịch",
    "mua",
    "mùa",
    "thang",
    "tháng",
    "sap",
    "sắp",
    "toi",
    "tới",
    "va",
    "và",
    "co",
    "có",
    "ngan",
    "ngân",
    "sach",
    "sách",
    "gia",
    "giá",
    "re",
    "rẻ",
    "khoang",
    "khoảng",
    "duoi",
    "dưới",
    "tren",
    "trên",
    "nen",
    "nên",
    "dau",
    "đâu",
    "nao",
    "nào",
    "choi",
    "chơi",
    "tam",
    "tắm",
    "nhung",
    "nhưng",
    "chi",
    "chỉ",
    "tr",
    "trieu",
    "triệu",
    "k",
    "vnd",
    "dong",
    "đồng",
    "nghin",
    "nghìn",
    "ngan",
    "ngàn",
}


def _expand_month_range(start: Optional[int], end: Optional[int]) -> set[int]:
    if start is None or end is None:
        return set()

    if start <= end:
        return set(range(start, end + 1))

    return set(range(start, 13)) | set(range(1, end + 1))


def _months_overlap(
    src_start: Optional[int],
    src_end: Optional[int],
    target_start: Optional[int],
    target_end: Optional[int],
) -> bool:
    if target_start is None or target_end is None:
        return True

    src_months = _expand_month_range(src_start, src_end)
    target_months = _expand_month_range(target_start, target_end)

    if not src_months:
        return False

    return len(src_months.intersection(target_months)) > 0


def _extract_query_terms(query: str) -> List[str]:
    if not query:
        return []

    terms = re.findall(r"[\wÀ-ỹ]+", query.lower())
    return [
        term
        for term in terms
        if len(term) >= 2
        and term not in STOPWORDS
        and not term.isdigit()
        and not re.search(r"\d", term)
    ]


def _text_match(query_terms: List[str], haystack: str) -> bool:
    if not query_terms:
        return True
    return any(term in haystack for term in query_terms)


def _count_term_matches(query_terms: List[str], haystack: str) -> int:
    if not query_terms:
        return 0
    return sum(1 for term in query_terms if term in haystack)


def _is_marine_query(query: str) -> bool:
    text = query.lower()
    marine_patterns = [
        "đi biển",
        "tam bien",
        "tắm biển",
        "bien",
        "bãi biển",
        "ngắm biển",
        "lặn",
        "san hô",
    ]
    return any(pattern in text for pattern in marine_patterns)


def _marine_score(haystack: str) -> int:
    positive_keywords = [
        "tắm biển",
        "bãi biển",
        "vịnh",
        "đảo",
        "san hô",
        "hải sản",
        "du thuyền",
        "kayak",
    ]
    negative_keywords = ["biển mây", "săn mây"]

    score = 0
    for keyword in positive_keywords:
        if keyword in haystack:
            score += 1
    for keyword in negative_keywords:
        if keyword in haystack:
            score -= 2
    return score


def search_destinations(
    query: str = "",
    month_start: Optional[int] = None,
    month_end: Optional[int] = None,
    max_budget: Optional[int] = None,
    destination: Optional[str] = None,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            location,
            season,
            description,
            activities,
            avg_cost,
            price_level,
            season_start,
            season_end,
            avg_cost_min,
            avg_cost_max
        FROM destinations
        """
    )

    rows = cur.fetchall()
    query_terms = _extract_query_terms(query)
    marine_query = _is_marine_query(query)
    
    staged_results = []

    for row in rows:
        location, season, description, activities, avg_cost, price_level = row[:6]
        season_start, season_end, avg_cost_min, avg_cost_max = row[6:]

        if destination:
            # Nếu khách đòi đi Đà Lạt, mà location (từ DB) không chứa chữ Đà Lạt thì bỏ qua luôn!
            if destination.strip().lower() not in location.lower():
                continue

        searchable_text = build_destination_semantic_text(
            location=location,
            description=description,
            activities=activities,
            price_level=price_level,
            season=season,
        )

        if not _text_match(query_terms, searchable_text):
            continue

        if not _months_overlap(season_start, season_end, month_start, month_end):
            continue

        if max_budget is not None:
            if avg_cost_min is None or avg_cost_min > max_budget:
                continue

        term_score = _count_term_matches(query_terms, searchable_text)
        budget_score = 0
        budget_match_tier = 0
        marine_bonus = _marine_score(searchable_text) if marine_query else 0

        if marine_query and marine_bonus <= 0:
            continue

        if max_budget is not None:
            if avg_cost_max is not None and avg_cost_max <= max_budget:
                budget_match_tier = 2
                budget_score = max_budget - avg_cost_max
            elif avg_cost_min is not None and avg_cost_min <= max_budget:
                budget_match_tier = 1
                budget_score = max_budget - avg_cost_min

        staged_results.append(
            {
                "candidate_id": build_destination_candidate_id(
                    location=location,
                    season=season,
                    description=description,
                    activities=activities,
                ),
                "semantic_text": searchable_text,
                "term_score": term_score,
                "marine_bonus": marine_bonus,
                "budget_match_tier": budget_match_tier,
                "budget_score": budget_score,
                "cost_sort": avg_cost_max if avg_cost_max is not None else 10**12,
                "result": {
                    "location": location,
                    "season": season,
                    "description": description,
                    "activities": activities,
                    "cost": avg_cost,
                    "price_level": price_level,
                    "season_start": season_start,
                    "season_end": season_end,
                    "avg_cost_min": avg_cost_min,
                    "avg_cost_max": avg_cost_max,
                },
            }
        )

    semantic_map = semantic_scores(
        query,
        {item["candidate_id"]: item["semantic_text"] for item in staged_results},
    )

    scored_results = []
    for item in staged_results:
        scored_results.append(
            (
                item["term_score"],
                semantic_map.get(item["candidate_id"], 0.0),
                item["marine_bonus"],
                item["budget_match_tier"],
                item["budget_score"],
                item["cost_sort"],
                item["result"],
            )
        )

    scored_results.sort(
        key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], item[5])
    )
    results = [item[6] for item in scored_results]

    cur.close()
    conn.close()

    return results
