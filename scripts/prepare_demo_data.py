"""
Prepare demo data for Text Drift → F1 Degradation → Recovery pipeline.
Splits reference_normal_v3.csv into train/val/test (80:10:10).
Converts all splits + current_trend_v3.csv to nlu.yml format.
"""
import csv, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RASA_DIR = os.path.join(os.path.dirname(__file__), "..", "rasa_bot")
CURRENT_NLU = os.path.join(RASA_DIR, "data", "train", "nlu.yml")

SPLIT_RATIO = (0.8, 0.1, 0.1)

def csv_to_nlu_rows(filepath):
    """Read CSV and group by intent. Returns {intent: [texts]}"""
    groups = {}
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row["intent"].strip()
            text = row["text"].strip()
            groups.setdefault(intent, []).append(text)
    return groups

def write_nlu_yaml(groups, out_path, description=""):
    """Write grouped intents to nlu.yml format."""
    lines = [f"# {description}" if description else "", 'version: "3.1"', "", "nlu:"]
    for intent in sorted(groups.keys()):
        texts = groups[intent]
        lines.append(f"  - intent: {intent}")
        lines.append('    examples: |')
        for t in texts:
            lines.append(f"      - {t}")
    content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  -> {out_path} ({sum(len(v) for v in groups.values())} examples, {len(groups)} intents)")

# ─── Step 1: Split reference_normal_v3.csv ───
print("=" * 60)
print("BUOC 1: TACH REFERENCE DATASET (80:10:10)")
print("=" * 60)

ref_path = os.path.join(DATA_DIR, "reference_normal_v3.csv")
ref_groups = csv_to_nlu_rows(ref_path)

train_groups, val_groups, test_groups = {}, {}, {}
for intent, texts in ref_groups.items():
    random.shuffle(texts)
    n = len(texts)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])
    train_groups[intent] = texts[:n_train]
    val_groups[intent] = texts[n_train:n_train + n_val]
    test_groups[intent] = texts[n_train + n_val:]

print(f"\n  Tong reference: {sum(len(v) for v in ref_groups.values())} examples")
print(f"  Train: {sum(len(v) for v in train_groups.values())} ({SPLIT_RATIO[0]*100:.0f}%)")
print(f"  Val:   {sum(len(v) for v in val_groups.values())} ({SPLIT_RATIO[1]*100:.0f}%)")
print(f"  Test:  {sum(len(v) for v in test_groups.values())} ({SPLIT_RATIO[2]*100:.0f}%)")

write_nlu_yaml(train_groups, os.path.join(DATA_DIR, "reference_train.yml"),
               "Reference training set (80% of reference_normal_v3)")
write_nlu_yaml(val_groups, os.path.join(DATA_DIR, "reference_val.yml"),
               "Reference validation set (10% of reference_normal_v3)")
write_nlu_yaml(test_groups, os.path.join(DATA_DIR, "reference_test.yml"),
               "Reference test set (10% of reference_normal_v3)")

# ─── Step 2: Convert current_trend_v3.csv to nlu.yml ───
print("\n" + "=" * 60)
print("BUOC 2: CHUYEN CURRENT TREND -> NLU.YML")
print("=" * 60)

cur_path = os.path.join(DATA_DIR, "current_trend_v3.csv")
cur_groups = csv_to_nlu_rows(cur_path)
write_nlu_yaml(cur_groups, os.path.join(DATA_DIR, "trend_test.yml"),
               "Trend test set (70% Ha Giang + 30% other)")

# ─── Step 3: Count original nlu.yml ───
print("\n" + "=" * 60)
print("BUOC 3: THONG KE CURRENT NLU.YML")
print("=" * 60)

if os.path.exists(CURRENT_NLU):
    with open(CURRENT_NLU, encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    ex_count = sum(1 for l in lines if l.strip().startswith("- ") and not l.strip().startswith("---"))
    intent_count = sum(1 for l in lines if l.strip().startswith("- intent:"))
    print(f"  Current nlu.yml: {ex_count} examples, {intent_count} intent blocks")
else:
    print(f"  WARNING: {CURRENT_NLU} not found!")

print("\n✅ Da chuan bi xong du lieu cho pipeline demo!")
print("\nFiles da tao:")
print("  data/reference_train.yml  - train set (80% reference)")
print("  data/reference_val.yml    - validation set (10% reference)")
print("  data/reference_test.yml   - test set (10% reference)")
print("  data/trend_test.yml       - test set (70% Ha Giang trend)")
