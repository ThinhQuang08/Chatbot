import os, sys, csv, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF_CSV = os.path.join(ROOT, "data", "reference_normal_v3.csv")
NLU_YML = os.path.join(ROOT, "rasa_bot", "data", "train", "nlu.yml")
OUTPUT = os.path.join(ROOT, "data", "reference_quality.csv")


def parse_nlu(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for item in data.get("nlu", []):
        intent = item.get("intent", "")
        for line in item.get("examples", "").strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                text = re.sub(r"\[([^\]]*)\]\{[^}]*\}", r"\1", line[2:].strip())
                rows.append({"text": text, "intent": intent})
    return pd.DataFrame(rows)


def main():
    print("=== Prepare reference_quality.csv ===")

    df_ref = pd.read_csv(REF_CSV)
    df_ref.columns = [c.strip().lower() for c in df_ref.columns]
    if "text" not in df_ref.columns:
        df_ref = df_ref.rename(columns={df_ref.columns[0]: "text"})
    df_ref = df_ref.drop_duplicates(subset=["text"])
    df_ref = df_ref[df_ref["text"].str.strip() != ""]
    print(f"  Reference CSV: {len(df_ref)} unique rows")

    df_human = parse_nlu(NLU_YML)
    df_human = df_human.drop_duplicates(subset=["text"])
    df_human = df_human[df_human["text"].str.strip() != ""]
    print(f"  Human NLU:     {len(df_human)} unique examples")

    combined = pd.concat([df_ref, df_human], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined[combined["text"].str.strip() != ""]
    combined = combined.reset_index(drop=True)

    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  → {OUTPUT} ({len(combined)} unique rows)")


if __name__ == "__main__":
    main()
