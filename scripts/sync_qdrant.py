# scripts/sync_qdrant.py
import json
import argparse
from typing import Dict, List

from database.db_connection import get_connection

from services.semantic_service import (
    build_destination_candidate_id,
    build_destination_semantic_text,
    sync_qdrant_documents,
)

def load_destination_documents() -> List[Dict[str, object]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
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
    cur.close()
    conn.close()

    documents: List[Dict[str, object]] = []
    for row in rows:
        dest_id = row[0]
        location, season, description, activities, avg_cost, price_level = row[1:7]
        season_start, season_end, avg_cost_min, avg_cost_max = row[7:]

        candidate_id = build_destination_candidate_id(
            location=location,
            season=season,
            description=description,
            activities=activities,
        )
        semantic_text = build_destination_semantic_text(
            location=location,
            description=description,
            activities=activities,
            price_level=price_level,
            season=season,
        )

        documents.append(
            {
                "candidate_id": candidate_id,
                "text": semantic_text,
                "payload": {
                    "destination_id": dest_id,
                    "candidate_id": candidate_id,
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

    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate Qdrant collection before syncing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of points per upsert batch.",
    )
    args = parser.parse_args()

    documents = load_destination_documents()
    result = sync_qdrant_documents(
        documents=documents,
        recreate_collection=args.recreate,
        batch_size=args.batch_size,
        collection_name="travel_destinations"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
