#database/import_data.py
import json
from database.db_connection import get_connection


INPUT_FILE = "data/data_normalized_destination.json"


def to_legacy_season(start: int | None, end: int | None) -> str | None:
    if start is None or end is None:
        return None
    return f"Tháng {start} - Tháng {end}"


def to_legacy_avg_cost(min_cost: int | None, max_cost: int | None) -> str | None:
    if min_cost is None or max_cost is None:
        return None
    return f"{min_cost:,} - {max_cost:,} VNĐ".replace(",", ".")


def main() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS destinations;")
    cur.execute(
        """
        CREATE TABLE destinations (
            id SERIAL PRIMARY KEY,
            location TEXT NOT NULL,
            season TEXT,
            description TEXT NOT NULL,
            activities TEXT,
            avg_cost TEXT,
            price_level TEXT NOT NULL,
            season_start INTEGER,
            season_end INTEGER,
            avg_cost_min INTEGER,
            avg_cost_max INTEGER,
            activities_list TEXT[] NOT NULL DEFAULT '{}'
        );
        """
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        activities_list = item.get("activities", [])
        activities_text = ", ".join(activities_list)
        season_start = item.get("season_start")
        season_end = item.get("season_end")
        avg_cost_min = item.get("avg_cost_min")
        avg_cost_max = item.get("avg_cost_max")

        cur.execute(
            """
            INSERT INTO destinations (
                location,
                season,
                description,
                activities,
                avg_cost,
                price_level,
                season_start,
                season_end,
                avg_cost_min,
                avg_cost_max,
                activities_list
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                item["location"],
                to_legacy_season(season_start, season_end),
                item["description"],
                activities_text,
                to_legacy_avg_cost(avg_cost_min, avg_cost_max),
                item["price_level"],
                season_start,
                season_end,
                avg_cost_min,
                avg_cost_max,
                activities_list,
            ),
        )

    conn.commit()
    print(f"Imported {len(data)} normalized records into destinations")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
