# Evaluate current model on reference_normal_v3.csv baseline data.
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
RASA_BIN = os.path.join(ROOT_DIR, ".venv", "bin", "rasa")

_ENV = os.environ.copy()
_PYTHONPATH = _ENV.get("PYTHONPATH", "")
if ROOT_DIR not in _PYTHONPATH.split(os.pathsep):
    _ENV["PYTHONPATH"] = os.pathsep.join(filter(None, [_PYTHONPATH, ROOT_DIR]))

os.makedirs(REPORT_DIR, exist_ok=True)


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
        if intent not in groups:
            groups[intent] = []
        groups[intent].append(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("version: \"3.1\"\n\nnlu:\n")
        for intent in sorted(groups.keys()):
            texts = groups[intent]
            unique = list(dict.fromkeys(texts))
            f.write(f"  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for t in unique:
                escaped = t.replace("\"", "\\\"")
                f.write(f"      - \"{escaped}\"\n")
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
        print("No trained model found in rasa_bot/models/")
        sys.exit(1)
    return models[-1]


def main():
    csv_path = os.path.join(ROOT_DIR, "data", "reference_normal_v3.csv")
    baseline_nlu = os.path.join(RASA_DIR, "data", "test", "baseline_nlu.yml")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("  BASELINE EVALUATION — reference_normal_v3.csv")
    print("=" * 60)

    n_total = convert_csv_to_nlu(csv_path, baseline_nlu)

    model_path = find_best_model()
    print(f"\nModel: {os.path.basename(model_path)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = subprocess.run(
        [
            RASA_BIN, "test", "nlu",
            "--nlu", "data/test/baseline_nlu.yml",
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

    print(f"\n  Accuracy: {report.get('accuracy', 0):.4f}")

    # Compare with known best F1
    best_path = os.path.join(RASA_DIR, "best_f1_score.txt")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_f1 = float(f.read().strip())
        print(f"  Best recorded macro F1 (held-out test): {best_f1:.4f}")
        print(f"  Baseline macro F1 (reference_normal_v3): {macro.get('f1-score', 0):.4f}")
        diff = macro.get("f1-score", 0) - best_f1
        print(f"  Difference: {diff:+.4f} ({diff/best_f1*100:+.1f}%)")

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

    # Save report
    txt_path = os.path.join(REPORT_DIR, "baseline_evaluation_report.txt")
    with open(txt_path, "w") as outf:
        outf.write(result.stdout)
        outf.write("\n\n")
        json.dump(report, outf, indent=2)
    print(f"\nReport saved: {txt_path}")
    print(f"Test file: {baseline_nlu}")


if __name__ == "__main__":
    main()
