import json
import re

INPUT_FILE = "data/destination.json"
OUTPUT_FILE = "data/data_normalized_destination.json"


def parse_cost(cost_text):
    numbers = re.findall(r'\d[\d\.]*', cost_text)

    if len(numbers) >= 2:
        min_cost = int(numbers[0].replace(".", ""))
        max_cost = int(numbers[1].replace(".", ""))
    else:
        min_cost = max_cost = int(numbers[0].replace(".", ""))

    return min_cost, max_cost


def parse_season(season_text):
    months = re.findall(r'\d+', season_text)

    if len(months) >= 2:
        return int(months[0]), int(months[1])

    if len(months) == 1:
        m = int(months[0])
        return m, m

    return None, None


def parse_activities(activity_text):

    activities = activity_text.split(",")

    cleaned = []
    for a in activities:
        a = a.strip().lower()
        cleaned.append(a)

    return cleaned


def convert():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:

        min_cost, max_cost = parse_cost(item["Avg_Cost"])
        season_start, season_end = parse_season(item["Season"])

        new_item = {
            "location": item["Location"],
            "description": item["Description"],
            "activities": parse_activities(item["Activities"]),
            "season_start": season_start,
            "season_end": season_end,
            "avg_cost_min": min_cost,
            "avg_cost_max": max_cost,
            "price_level": item["Price_Level"].lower()
        }

        new_data.append(new_item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("Data converted successfully!")


if __name__ == "__main__":
    convert()
