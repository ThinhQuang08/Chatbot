# Evaluate baseline model on current_trend_v3 drift data.
import csv
import json
import os
import subprocess
import sys
import glob
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RASA_DIR = os.path.join(ROOT_DIR, "rasa_bot")
RESULTS_DIR = os.path.join(RASA_DIR, "results")
REPORT_DIR = os.path.join(ROOT_DIR, "report_output")

_ENV = os.environ.copy()
_PYTHONPATH = _ENV.get("PYTHONPATH", "")
if ROOT_DIR not in _PYTHONPATH.split(os.pathsep):
    _ENV["PYTHONPATH"] = os.pathsep.join(filter(None, [_PYTHONPATH, ROOT_DIR]))

os.makedirs(REPORT_DIR, exist_ok=True)

INTENT_MAP = {
    "search_destination": "travel_planning",
    "search_travel": "travel_planning",
}


def convert_csv_to_nlu(csv_path, output_path):
    seen = set()
    intent_counts = Counter()
    total = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups = {}
    for row in rows:
        text = row["text"].strip()
        intent = row["intent"].strip()
        intent = INTENT_MAP.get(intent, intent)

        if intent not in groups:
            groups[intent] = []
        groups[intent].append(text)

    with open(output_path, "w") as f:
        f.write("version: \"3.1\"\n\nnlu:\n")
        for intent in sorted(groups.keys()):
            texts = groups[intent]
            unique = list(dict.fromkeys(texts))  # preserve order, deduplicate
            f.write(f"  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for t in unique:
                escaped = t.replace('"', '\\"')
                f.write(f"      - {escaped}\n")
            intent_counts[intent] = len(unique)
            total += len(unique)

    print(f"Converted {total} examples ({len(groups)} intents) to {output_path}")
    for intent, count in intent_counts.most_common():
        print(f"  {intent:30s} {count}")
    return total


def find_best_model():
    models = sorted(
        glob.glob(os.path.join(RASA_DIR, "models", "*.tar.gz")),
        key=os.path.getctime,
    )
    if not models:
        print("No trained model found.")
        sys.exit(1)
    return models[-1]


def main():
    csv_path = os.path.join(ROOT_DIR, "data", "current_trend_v3.csv")
    drift_nlu = os.path.join(RASA_DIR, "data", "test", "drift_nlu.yml")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("  DRIFT EVALUATION — current_trend_v3")
    print("=" * 60)

    n_total = convert_csv_to_nlu(csv_path, drift_nlu)

    model_path = find_best_model()
    print(f"\nModel: {os.path.basename(model_path)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for f in os.listdir(RESULTS_DIR):
        fp = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    result = subprocess.run(
        [
            "rasa", "test", "nlu",
            "--nlu", "data/test/drift_nlu.yml",
            "--model", model_path,
            "--out", RESULTS_DIR,
        ],
        cwd=RASA_DIR, capture_output=True, text=True, env=_ENV,
    )

    if result.returncode != 0:
        print(f"Error:\n{result.stderr[:1000]}")
        sys.exit(1)

    report_path = os.path.join(RESULTS_DIR, "intent_report.json")
    if not os.path.exists(report_path):
        print("intent_report.json not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    print(f"\n{'Intent':28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  {'Support':>7s}")
    print("-" * 56)
    for k in sorted(report):
        if k in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            continue
        v = report[k]
        print(f"{k:28s} {v['precision']:6.3f} {v['recall']:6.3f} "
              f"{v['f1-score']:6.3f}  {v['support']:7d}")
    print("-" * 56)
    print(f"{'macro avg':28s} {macro.get('precision', 0):6.3f} "
          f"{macro.get('recall', 0):6.3f} "
          f"{macro.get('f1-score', 0):6.3f}  {n_total:7d}")
    print(f"{'weighted avg':28s} {weighted.get('precision', 0):6.3f} "
          f"{weighted.get('recall', 0):6.3f} "
          f"{weighted.get('f1-score', 0):6.3f}  {n_total:7d}")

    # Compare with held-out test
    print(f"\n  vs held-out test set (297 ex): macro F1 = 0.6948")
    print(f"  Drift macro F1:                       {macro.get('f1-score', 0):.4f}")
    print(f"  Drift accuracy:                       {report.get('accuracy', 0):.4f}")
    drop = 0.6948 - macro.get("f1-score", 0)
    print(f"  Performance drop:                     {drop:.4f} ({drop/0.6948*100:.1f}%)")

    txt_path = os.path.join(REPORT_DIR, "drift_evaluation_report.txt")
    with open(txt_path, "w") as outf:
        outf.write(result.stdout)
        outf.write("\n\n")
        json.dump(report, outf, indent=2)

    # Confusion matrix
    errors_path = os.path.join(RESULTS_DIR, "intent_errors.json")
    if os.path.exists(errors_path):
        with open(errors_path) as f:
            errors = json.load(f)
        confusions = Counter(
            (e["intent"], e["intent_prediction"]["name"]) for e in errors
        )
        print(f"\nTop confusions:")
        print(f"  {'True':28s} {'→ Predicted':28s}  Count")
        print(f"  {'-'*28}   {'-'*28}  {'-'*5}")
        for (true, pred), cnt in confusions.most_common(15):
            print(f"  {true:28s}   {pred:28s}  {cnt:5d}")

    print(f"\nReport saved: {txt_path}")
    print(f"Drift test file: {drift_nlu}")


if __name__ == "__main__":
    main()
