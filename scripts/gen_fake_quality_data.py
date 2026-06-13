import os, sys, csv, random, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF_PATH = os.path.join(ROOT, "data", "reference_quality.csv")
OUTPUT = os.path.join(ROOT, "data", "quality_fake_current.csv")

RANDOM_SEED = 42
N_DAYS = 10
ROWS_PER_DAY = 300
TOTAL = N_DAYS * ROWS_PER_DAY


def degrade(text, level):
    level = min(level, 4)
    if level == 0:
        return text

    chars = list(text)

    if level >= 1 and random.random() < 0.15:
        chars = [c for c in chars if not unicodedata_category(c) in ("Mn", "Mc")]

    if level >= 2 and random.random() < 0.10:
        vowels = "aeiouyAEIOUY"
        chars = [c for c in chars if c not in vowels or random.random() > 0.5]

    if level >= 3 and random.random() < 0.08:
        chars = [c for c in chars if c.isalnum() or c.isspace()]

    return "".join(chars)


def unicodedata_category(ch):
    import unicodedata
    return unicodedata.category(ch)


def apply_corruption(text, day, row_idx):
    idx = day - 1
    if idx < 3:
        return text

    if idx >= 7 and random.random() < 0.30:
        return " "  # single space = will be detected as empty by is_empty feature

    if idx >= 5 and random.random() < 0.20:
        words = text.split()
        return " ".join(words[:max(1, len(words)//3)])

    if idx >= 4 and random.random() < 0.15:
        emoji_map = {
            "chụp ảnh": "📷", "ăn": "🍜", "biển": "🌊",
            "ngủ": "😴", "đẹp": "✨", "vui": "🎉",
        }
        for kw, em in emoji_map.items():
            if kw in text.lower():
                text = re.sub(rf"\b{kw}\b", em, text, flags=re.IGNORECASE, count=1)
                break
        return text

    if idx >= 6 and random.random() < 0.25:
        abbr = {"không": "ko", "được": "dc", "nhưng": "nhg",
                "gì": "j", "rồi": "r", "vậy": "z"}
        for word, ab in abbr.items():
            text = re.sub(rf"\b{word}\b", ab, text, count=1)
        return text

    if idx >= 3 and random.random() < 0.10:
        text = text.lower()
        result = []
        for ch in text:
            if unicodedata_category(ch) in ("Mn", "Mc"):
                continue
            result.append(ch)
        return "".join(result)

    return text


def main():
    random.seed(RANDOM_SEED)

    print("=== Generate fake quality data ===")
    df = pd.read_csv(REF_PATH)
    source_texts = df["text"].dropna().str.strip().tolist()
    source_texts = [t for t in source_texts if len(t) > 5]
    print(f"  Source pool: {len(source_texts)} texts")

    rows = []
    for day in range(1, N_DAYS + 1):
        for ri in range(ROWS_PER_DAY):
            text = random.choice(source_texts)
            text = apply_corruption(text, day, ri)
            rows.append({"date": f"2026-06-{day:02d}", "text": text, "day": day})

    df_out = pd.DataFrame(rows)
    # Keep duplicates — they represent real-world duplicate queries
    print(f"  Generated {len(df_out)} rows")

    print("  Per-day stats:")
    for day in range(1, N_DAYS + 1):
        subset = df_out[df_out["day"] == day]
        empty_pct = (subset["text"].str.strip() == "").mean() * 100
        short_pct = (subset["text"].str.len() < 5).mean() * 100
        print(f"    Day {day}: {len(subset)} rows, empty={empty_pct:.0f}%, short={short_pct:.0f}%")

    df_out = df_out.drop(columns=["day"])
    df_out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  → {OUTPUT} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
