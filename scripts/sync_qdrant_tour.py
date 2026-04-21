# scripts/sync_qdrant_tour.py
import json
import argparse
from typing import Dict, List

from database.db_connection import get_connection
from services.semantic_service import (
    build_tour_candidate_id,
    build_tour_semantic_text,
    sync_qdrant_documents,
)

def load_tour_documents() -> List[Dict[str, object]]:
    print("Đang truy vấn bảng 'tours' từ PostgreSQL...")
    conn = get_connection()
    cur = conn.cursor()

    # Query toàn bộ thông tin tour ra
    cur.execute("""
        SELECT 
            t.tour_id, 
            t.tour_name, 
            t.departure, 
            -- Cột 4: Gom các điểm đến thành JSON array
            COALESCE(json_agg(d.location) FILTER (WHERE d.location IS NOT NULL), '[]'::json) AS destinations,
            t.price, 
            t.currency, 
            t.days,
            t.nights,
            t.duration_text, 
            t.transportation, 
            t.rag_knowledge_base,
            t.season_tag,
            t.data_source
        FROM tours t
        LEFT JOIN tour_destinations td ON t.tour_id = td.tour_id
        LEFT JOIN destinations d ON td.destination_id = d.id
        -- WHERE t.is_vectorized = FALSE
        GROUP BY t.tour_id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    documents: List[Dict[str, object]] = []

    for row in rows:
        (tour_id, tour_name, departure, destination_json, price, currency,
         days, nights, duration_text, transportation, rag_kb_json,
         season_tag, data_source) = row

        # Xử lý trường JSONB từ PostgreSQL (Thường psycopg2 tự parse thành dict/list, nhưng ta cứ check cho chắc)
        destination = destination_json if isinstance(destination_json, list) else json.loads(destination_json)
        rag_kb = rag_kb_json if isinstance(rag_kb_json, dict) else json.loads(rag_kb_json)
        
        # Lấy một đoạn mô tả (từ ngày 1) để tạo text ngữ nghĩa
        itinerary = rag_kb.get("itinerary", [])
        overview_desc = itinerary[0].get("description", "") if itinerary else ""

        # --- TẠO ID VÀ TEXT ---
        chunk_type = "overview"
        candidate_id = build_tour_candidate_id(tour_id=tour_id, chunk_type=chunk_type)
        semantic_text = build_tour_semantic_text(
            tour_name=tour_name, 
            destinations=destination, 
            description=overview_desc
        )

        # --- ĐÓNG GÓI PAYLOAD  ---
        payload = {
            "tour_id": tour_id,
            "tour_name": tour_name,
            "departure": departure,
            "destination": destination,
            "price": price,
            "currency": currency,
            "days": days,
            "nights": nights,
            "duration_text": duration_text,
            "transportation": transportation,
            "type": chunk_type,
            "season_tag": season_tag,
            "data_source": data_source,
            "semantic_text": semantic_text 
        }

        documents.append({
            "candidate_id": candidate_id,
            "text": semantic_text,
            "payload": payload
        })

    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa và tạo lại Qdrant collection trước khi đồng bộ.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Số lượng points mỗi lần upsert.",
    )
    args = parser.parse_args()

    documents = load_tour_documents()
    
    if not documents:
        print("Không có tài liệu nào để đồng bộ.")
        return

    print(f"Bắt đầu đồng bộ {len(documents)} bản ghi Tour lên Qdrant...")
    result = sync_qdrant_documents(
        documents=documents,
        recreate_collection=args.recreate,
        batch_size=args.batch_size,
        collection_name="travel_tours"
    )
    
    print("\nKẾT QUẢ ĐỒNG BỘ QDRANT:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()