import argparse
import random
import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import get_connection

INTENTS_POOL = {
    "search_travel": 0.14,
    "inform": 0.11,
    "search_activity": 0.10,
    "greet": 0.06,
    "search_destination": 0.06,
    "affirm": 0.06,
    "search_accommodation": 0.06,
    "search_price": 0.05,
    "goodbye": 0.05,
    "search_food_dining": 0.05,
    "ask_itinerary": 0.05,
    "ask_policy_booking": 0.05,
    "ask_tour_info": 0.05,
    "ask_transportation": 0.04,
    "out_of_scope": 0.02,
    "ask_location_feature": 0.03,
    "book_tour": 0.02,
    "deny": 0.02,
    "bot_challenge": 0.01,
}

DRIFT_INTENTS = {"out_of_scope": 0.30, "nlu_fallback": 0.25, "search_travel": 0.20,
                  "search_price": 0.15, "bot_challenge": 0.10}

DESTINATIONS = ["Đà Lạt", "Sapa", "Phú Quốc", "Nha Trang", "Đà Nẵng",
                "Vũng Tàu", "Hà Nội", "Hà Giang", "Hội An", "Huế"]

NEW_DESTINATIONS = ["Măng Đen", "Tà Xùa", "Phú Quý", "Trị An", "Tà Năng",
                    "Pù Luông", "Mộc Châu", "Cao Bằng"]

DEPARTURES = ["Sài Gòn", "Hà Nội", "Đà Nẵng", "Vinh"]

ACTIVITIES = ["leo núi", "trekking", "lặn san hô", "cắm trại", "đạp xe",
              "dù lượn", "chèo thuyền"]

NEW_VOCAB = ["glamping", "săn mây", "chữa lành", "team building", "staycation"]

BUDGETS_NORMAL = [2000000, 3000000, 5000000, 8000000, 10000000, 15000000, 20000000]
BUDGETS_LOW    = [200000, 300000, 500000, 800000, 1000000, 1500000, 2000000]

CONFIDENCE_HIGH = (0.85, 0.99)
CONFIDENCE_LOW  = (0.30, 0.60)

TEMPLATES = {
    "search_travel": [
        "tôi muốn đi {dest}",
        "cho mình hỏi tour {dest}",
        "đi {dest} chi phí thế nào",
        "muốn đi du lịch {dest} {budget_text}",
        "từ {dep} đi {dest} bằng xe gì",
        "tìm tour {dest} {dur}",
        "có tour nào đi {dest} không",
        "mình cần tìm tour {dest} giá rẻ",
        "đi {dest} mùa nào đẹp",
        "cho em xin tour {dest} {budget_text}",
    ],
    "inform": [
        "{dest}",
        "khoảng {dur}",
        "tầm {budget_text}",
        "đi {trans} nhé",
        "{month}",
        "{activity}",
    ],
    "search_activity": [
        "muốn {activity} ở {dest}",
        "có chỗ nào {activity} không",
        "tìm chỗ {activity}",
        "đi {activity} ở đâu",
        "cho thuê xe {activity}",
    ],
    "search_destination": [
        "gợi ý điểm du lịch",
        "có nơi nào đẹp để đi chơi không",
        "muốn đi du lịch {month}",
        "{month} đi đâu đẹp",
        "nên đi đâu chơi",
    ],
    "search_accommodation": [
        "có {cat} nào ở {dest} không",
        "tìm {cat} giá rẻ {dest}",
        "cho mình xin {cat} {dest}",
        "ở {dest} có {cat} nào tốt",
    ],
    "search_price": [
        "giá tour này bao nhiêu",
        "đi {dest} chi phí bao nhiêu",
        "du lịch tiết kiệm",
        "tour giá rẻ",
        "chi phí tour {dest}",
    ],
    "search_food_dining": [
        "{dest} có đặc sản gì",
        "ăn gì ở {dest}",
        "có quán nào ngon ở {dest}",
        "đồ ăn {dest} thế nào",
    ],
    "ask_tour_info": [
        "lịch trình tour {dest}",
        "tour này có hdv không",
        "thông tin chi tiết tour {dest}",
        "bao gồm những gì",
    ],
    "ask_transportation": [
        "đi {dest} bằng gì",
        "có xe khách đi {dest} không",
        "phương tiện đi {dest}",
    ],
    "ask_location_feature": [
        "{dest} có gì đẹp",
        "{dest} mùa nào đẹp nhất",
        "{dest} có gì chơi",
    ],
    "ask_policy_booking": [
        "chính sách hủy tour",
        "hoàn tiền thế nào",
        "có được hoàn không",
    ],
    "ask_itinerary": [
        "gợi ý lịch trình {dest} {dur}",
        "đi {dest} {dur} nên đi đâu",
    ],
    "book_tour": [
        "đặt tour {dest}",
        "muốn đặt tour {dest}",
        "đăng ký tour {dest}",
    ],
    "greet": ["chào bạn", "xin chào", "hey", "hi", "alo"],
    "goodbye": ["tạm biệt", "cảm ơn", "bye", "ok cảm ơn"],
    "affirm": ["có", "được", "ok", "ừ", "ok bạn"],
    "deny": ["không", "không có", "không phải", "thôi"],
    "out_of_scope": [
        "thời tiết hôm nay thế nào",
        "mở cửa hàng lúc mấy giờ",
        "cho mình hỏi số điện thoại",
        "bạn có biết nấu ăn không",
        "giải bài tập này giúp tôi",
    ],
    "bot_challenge": [
        "bạn là ai",
        "bạn là người hay máy",
        "ai lập trình ra bạn",
        "bạn tên gì",
    ],
}

DRIFT_CONCEPT_TEMPLATES = {
    "search_travel": [
        "cho mình xin info tour {new_dest}",
        "muốn đi {new_dest} {dur} chi phí sao",
        "có ai đi {new_dest} chưa review giúp",
        "tìm tour {new_dest} giá sinh viên",
        "mình muốn {concept} ở {new_dest}",
        "có tour {concept} nào không",
        "group {people} người muốn đi {concept} {new_dest}",
        "tư vấn {concept} {new_dest} {dur}",
    ],
    "search_activity": [
        "muốn đi {concept} ở {new_dest}",
        "có {concept} nào gần đây không",
        "chi phí {concept} {new_dest}",
        "thuê đồ {concept} ở đâu",
    ],
    "search_price": [
        "giá {concept} {new_dest} bao nhiêu",
        "chi phí {concept} cho nhóm {people} người",
        "bảng giá tour {concept}",
    ],
}

MISC_NOISE = ["ạ", "nhé", "nha", "shop", "bot ơi", "tư vấn với", "gấp ạ"]


def pick_random(values):
    return random.choice(values)


def weighted_intent(weights):
    intents, probs = zip(*weights.items())
    return random.choices(intents, weights=probs, k=1)[0]


def interpolate(start, end, t):
    return start + (end - start) * t


def generate_sentence(intent, drift_scale, use_new_dest):
    pool = TEMPLATES.get(intent, ["{dest}"])
    template = pick_random(pool)
    kwargs = dict(
        dest=random.choice(NEW_DESTINATIONS if use_new_dest and random.random() < drift_scale else DESTINATIONS),
        new_dest=random.choice(NEW_DESTINATIONS),
        dep=random.choice(DEPARTURES),
        activity=random.choice(NEW_VOCAB if use_new_dest and random.random() < drift_scale else ACTIVITIES),
        concept=pick_random(NEW_VOCAB),
        cat=pick_random(CATEGORIES := ["khách sạn", "homestay", "resort", "villa", "nhà nghỉ"]),
        trans=pick_random(["máy bay", "xe khách", "tàu hỏa", "xe máy"]),
        month=f"tháng {random.randint(1, 12)}",
        dur=f"{random.randint(2, 5)} ngày {random.randint(1, 4)} đêm",
        budget_text=pick_random([f"{v // 1000000} triệu" for v in BUDGETS_NORMAL if v >= 1000000]
                                 + [f"{v // 1000}k" for v in BUDGETS_NORMAL]),
        people=random.randint(2, 15),
    )
    text = template.format(**kwargs)
    if random.random() < 0.3:
        text += " " + pick_random(MISC_NOISE)
    return text


def generate_row(intent, drift_scale, timestamp, use_new_dest, use_low_budget, use_low_conf):
    text = generate_sentence(intent, drift_scale, use_new_dest)

    budget = None
    if intent in {"search_travel", "inform", "search_price", "search_accommodation",
                   "ask_tour_info", "ask_itinerary"}:
        pool_bud = BUDGETS_LOW if use_low_budget else BUDGETS_NORMAL
        budget = pick_random(pool_bud) if random.random() < 0.7 else None

    confidence = round(random.uniform(*CONFIDENCE_LOW if use_low_conf else CONFIDENCE_HIGH), 3)

    dest_val = None
    for kw in NEW_DESTINATIONS + DESTINATIONS:
        if kw.lower() in text.lower():
            dest_val = kw
            break

    no_results = 1 if (use_low_budget and random.random() < 0.3) else 0

    return dict(
        session_id=f"drift_{random.randint(10000, 99999)}_{random.randint(0, 999)}",
        raw_text=text,
        predicted_intent=intent,
        confidence_score=confidence,
        destination=dest_val,
        parsed_budget=budget,
        no_results_flag=no_results,
        timestamp=timestamp,
    )


def build_data(volume, drift_types, severity, mode, drift_ratio=0.6):
    ref_count = int(volume * (1 - drift_ratio))
    drift_count = volume - ref_count
    rows = []

    now = datetime.now()
    start_time = now - timedelta(days=30)
    ref_end = start_time + timedelta(days=int(30 * (1 - drift_ratio)))

    enable_intent_drift = "intent" in drift_types or "all" in drift_types
    enable_entity_drift = "entity" in drift_types or "all" in drift_types
    enable_budget_drift = "budget" in drift_types or "all" in drift_types
    enable_conf_drift = "confidence" in drift_types or "all" in drift_types
    enable_concept_drift = "concept" in drift_types or "all" in drift_types
    enable_no_results = "no_results" in drift_types or "all" in drift_types

    def get_normal_intent():
        return weighted_intent(INTENTS_POOL)

    def get_drift_intent(progress):
        if enable_intent_drift and random.random() < progress * severity:
            return weighted_intent(DRIFT_INTENTS)
        return get_normal_intent()

    # ── Reference period (no drift) ──
    for i in range(ref_count):
        ts = start_time + (ref_end - start_time) * (i / max(ref_count - 1, 1))
        intent = get_normal_intent()
        rows.append(generate_row(intent, 0.0, ts, False, False, False))

    # ── Drift period (progressive) ──
    drift_start = timestamps = [ref_end + (now - ref_end) * (i / max(drift_count - 1, 1))
                                 for i in range(drift_count)]

    for i in range(drift_count):
        ts = drift_start[i]
        if mode == "sudden":
            progress = 1.0 if i > drift_count // 3 else 0.0
        else:
            progress = i / max(drift_count - 1, 1)

        scale = progress * severity
        intent = get_drift_intent(progress)

        use_new_dest = enable_entity_drift and scale > 0
        use_low_budget = enable_budget_drift and random.random() < scale
        use_low_conf = enable_conf_drift and random.random() < scale * 1.2

        # For concept drift, use concept templates in later stages
        if enable_concept_drift and scale > 0.3 and random.random() < scale:
            concept_pool = DRIFT_CONCEPT_TEMPLATES.get(intent, None)
            if concept_pool is None and intent in DRIFT_INTENTS:
                concept_pool = DRIFT_CONCEPT_TEMPLATES["search_travel"]
            if concept_pool:
                template = pick_random(concept_pool)
                kwargs = dict(
                    new_dest=pick_random(NEW_DESTINATIONS),
                    concept=pick_random(NEW_VOCAB),
                    dur=f"{random.randint(2, 5)} ngày {random.randint(1, 4)} đêm",
                    people=random.randint(2, 15),
                    budget_text=f"{random.choice(BUDGETS_LOW) // 1000}k",
                )
                text = template.format(**kwargs)
                budget = pick_random(BUDGETS_LOW) if random.random() < 0.7 else None
                dest_val = pick_random(NEW_DESTINATIONS)
                confidence = round(random.uniform(*CONFIDENCE_LOW if use_low_conf else CONFIDENCE_HIGH), 3)
                no_results = 1 if enable_no_results and random.random() < scale else 0
                rows.append(dict(
                    session_id=f"drift_{random.randint(10000, 99999)}_{random.randint(0, 999)}",
                    raw_text=text, predicted_intent=intent,
                    confidence_score=confidence, destination=dest_val,
                    parsed_budget=budget, no_results_flag=no_results,
                    timestamp=ts,
                ))
                continue

        row = generate_row(intent, scale, ts, use_new_dest, use_low_budget, use_low_conf)
        if enable_no_results and random.random() < scale:
            row["no_results_flag"] = 1
        rows.append(row)

    return rows


def output_db(rows):
    conn = get_connection()
    cur = conn.cursor()
    batch = []
    for r in rows:
        batch.append((
            r["session_id"], r["raw_text"], r["predicted_intent"],
            r["confidence_score"], r["destination"], r["parsed_budget"],
            r["no_results_flag"], r["timestamp"],
        ))
    cur.executemany("""
        INSERT INTO ai_chat_analytics
            (session_id, raw_text, predicted_intent, confidence_score,
             destination, parsed_budget, no_results_flag, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, batch)
    conn.commit()
    cur.close()
    conn.close()


def output_csv(rows, ref_out, drift_out):
    df = pd.DataFrame(rows)
    ref_count = int(len(df) * 0.4)
    df["segment_label"] = ["reference"] * ref_count + ["drift"] * (len(df) - ref_count)
    df["timestamp"] = df["timestamp"].astype(str)
    ref_df = df[df["segment_label"] == "reference"].copy()
    drift_df = df[df["segment_label"] == "drift"].copy()
    ref_df.to_csv(ref_out, index=False, encoding="utf-8-sig")
    drift_df.to_csv(drift_out, index=False, encoding="utf-8-sig")
    df.to_csv("data/drift_full_data.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drift data for testing")
    parser.add_argument("--drift-types", default="all",
                        help="Comma-separated: intent,entity,budget,confidence,concept,no_results,all")
    parser.add_argument("--severity", type=float, default=0.3, help="Drift severity 0.1-0.9")
    parser.add_argument("--mode", choices=["gradual", "sudden", "both"], default="gradual")
    parser.add_argument("--volume", type=int, default=3000, help="Total rows to generate")
    parser.add_argument("--output", choices=["db", "csv", "both"], default="both")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    drift_types = [t.strip() for t in args.drift_types.split(",")]

    print(f"[GENERATE] Drift types: {drift_types}")
    print(f"[GENERATE] Severity: {args.severity}, Mode: {args.mode}, Volume: {args.volume}")
    print(f"[GENERATE] Generating data...")

    rows = build_data(args.volume, drift_types, args.severity, args.mode)

    print(f"[GENERATE] {len(rows)} rows generated "
          f"(~{int(len(rows)*0.4)} reference, ~{len(rows)-int(len(rows)*0.4)} drift)")

    if args.output in ("db", "both"):
        print(f"[GENERATE] Inserting into PostgreSQL...")
        output_db(rows)
        print(f"[GENERATE] DB insert complete.")

    if args.output in ("csv", "both"):
        ref_out = "data/reference_period.csv"
        drift_out = "data/drift_period.csv"
        print(f"[GENERATE] Writing CSV: {ref_out}, {drift_out}")
        output_csv(rows, ref_out, drift_out)
        print(f"[GENERATE] CSV output complete.")

    # Distribution summary
    intents = [r["predicted_intent"] for r in rows]
    from collections import Counter
    summary = Counter(intents)
    print(f"\n[SUMMARY] Intent distribution:")
    for intent, count in summary.most_common():
        pct = count / len(intents) * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:25s} {count:5d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
