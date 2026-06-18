# Train a baseline Rasa NLU model on reference_normal_v3.csv, targeting macro F1 ~0.7.
import csv
import json
import os
import subprocess
import sys
import glob
import shutil
import random
from collections import Counter, OrderedDict
from datetime import datetime

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
random.seed(42)

TRAIN_DIR = os.path.join(RASA_DIR, "data", "train_baseline_ref")
TEST_DIR = os.path.join(RASA_DIR, "data", "test")
MODEL_NAME = "baseline_ref"
DOMAIN_PATH = os.path.join(RASA_DIR, "domain_baseline_ref.yml")
CONFIG_PATH = os.path.join(RASA_DIR, "config.yml")


def load_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratify_split(rows, test_frac=0.2):
    by_intent = OrderedDict()
    for r in rows:
        intt = r["intent"].strip()
        by_intent.setdefault(intt, []).append(r)
    train, test = [], []
    for intt, items in by_intent.items():
        random.shuffle(items)
        n_test = max(1, round(len(items) * test_frac))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def write_nlu(rows, path):
    groups = OrderedDict()
    for r in rows:
        intt = r["intent"].strip()
        text = r["text"].strip()
        groups.setdefault(intt, []).append(text)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("version: \"3.1\"\n\nnlu:\n")
        for intent in groups:
            f.write(f"  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for t in groups[intent]:
                escaped = t.replace("\"", "\\\"")
                f.write(f"      - \"{escaped}\"\n")


def write_domain(intents, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("version: \"3.1\"\n\n")
        f.write("intents:\n")
        for intt in sorted(intents):
            f.write(f"  - {intt}\n")
        f.write("\nentities: []\n\nslots: {}\n\nresponses:\n  utter_default:\n  - text: \"Tôi không hiểu yêu cầu của bạn.\"\n")
        f.write("\nsession_config:\n  session_expiration_time: 60\n  carry_over_slots_to_new_session: true\n")


def find_latest_model():
    models = sorted(
        glob.glob(os.path.join(RASA_DIR, "models", f"{MODEL_NAME}*.tar.gz")),
        key=os.path.getctime,
    )
    return models[-1] if models else None


def main():
    csv_path = os.path.join(ROOT_DIR, "data", "reference_normal_v3.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("  TRAIN BASELINE — reference_normal_v3.csv → macro F1 ~0.7")
    print("=" * 60)

    rows = load_csv(csv_path)
    print(f"\nLoaded {len(rows)} rows, {len(set(r['intent'].strip() for r in rows))} intents")

    train_rows, test_rows = stratify_split(rows, test_frac=0.2)
    print(f"Train: {len(train_rows)} rows")
    print(f"Test:  {len(test_rows)} rows")
    train_counts = Counter(r["intent"].strip() for r in train_rows)
    test_counts = Counter(r["intent"].strip() for r in test_rows)
    print(f"Train intents: {dict(train_counts)}")
    print(f"Test intents:  {dict(test_counts)}")

    intents = sorted(set(r["intent"].strip() for r in rows))

    train_nlu = os.path.join(TRAIN_DIR, "nlu.yml")
    test_nlu = os.path.join(RASA_DIR, "data", "test", "baseline_ref_test.yml")

    write_nlu(train_rows, train_nlu)
    write_nlu(test_rows, test_nlu)
    write_domain(intents, DOMAIN_PATH)
    print(f"\nWrote train: {train_nlu}  ({len(train_rows)} examples)")
    print(f"Wrote test:  {test_nlu}  ({len(test_rows)} examples)")
    print(f"Wrote domain: {DOMAIN_PATH}  ({len(intents)} intents)")

    # Clean old model for this name
    old = find_latest_model()
    if old:
        os.remove(old)
        print(f"Removed old model: {old}")

    print(f"\n{'='*60}")
    print("  Training...")
    print(f"{'='*60}")
    train_cmd = [
        RASA_BIN, "train", "nlu",
        "--nlu", train_nlu,
        "--config", CONFIG_PATH,
        "--domain", DOMAIN_PATH,
        "--out", os.path.join(RASA_DIR, "models"),
        "--fixed-model-name", MODEL_NAME,
    ]
    train_proc = subprocess.run(
        train_cmd, cwd=RASA_DIR,
        capture_output=True, text=True, env=_ENV,
        timeout=3600,
    )
    print(train_proc.stdout[-2000:] if len(train_proc.stdout) > 2000 else train_proc.stdout)
    if train_proc.stderr.strip():
        print("STDERR:", train_proc.stderr[-1000:])

    if train_proc.returncode != 0:
        print("Training failed!")
        sys.exit(1)

    model_path = find_latest_model()
    if not model_path:
        print("Model not found after training!")
        sys.exit(1)
    print(f"\nTrained model: {model_path}")

    # Cross-validation for realistic F1 estimate
    print(f"\n{'='*60}")
    print("  Cross-validation (3-fold) for realistic F1 estimate...")
    print(f"{'='*60}")
    cv_dir = os.path.join(RESULTS_DIR, "eval_baseline_ref_cv")
    if os.path.exists(cv_dir):
        shutil.rmtree(cv_dir)

    cv_cmd = [
        RASA_BIN, "test", "nlu",
        "--cross-validation",
        "--folds", "3",
        "--nlu", train_nlu,
        "--config", CONFIG_PATH,
        "--domain", DOMAIN_PATH,
        "--out", cv_dir,
    ]
    cv_proc = subprocess.run(
        cv_cmd, cwd=RASA_DIR,
        capture_output=True, text=True, env=_ENV,
        timeout=3600,
    )
    if cv_proc.stdout.strip():
        for line in cv_proc.stdout.split("\n"):
            if any(kw in line for kw in ("F1", "Precision", "Recall", "Accuracy", "weighted", "macro")):
                print(f"  {line.strip()}")

    # Also do held-out test for per-intent breakdown
    print(f"\n{'='*60}")
    print("  Held-out test (per-intent breakdown)...")
    print(f"{'='*60}")
    eval_dir = os.path.join(RESULTS_DIR, "eval_baseline_ref")
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    eval_cmd = [
        RASA_BIN, "test", "nlu",
        "--model", model_path,
        "--nlu", test_nlu,
        "--out", eval_dir,
    ]
    eval_proc = subprocess.run(
        eval_cmd, cwd=RASA_DIR,
        capture_output=True, text=True, env=_ENV,
        timeout=600,
    )

    report_path = os.path.join(eval_dir, "intent_report.json")
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        print(eval_proc.stdout[-1500:])
        sys.exit(1)

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
          f"{macro.get('f1-score', 0):6.3f}  {len(test_rows):7d}")
    print(f"{'weighted avg':28s} {weighted.get('precision', 0):6.3f} "
          f"{weighted.get('recall', 0):6.3f} "
          f"{weighted.get('f1-score', 0):6.3f}  {len(test_rows):7d}")
    print(f"{'Accuracy':28s} {report.get('accuracy', 0):6.4f}")

    # Save full report
    txt_path = os.path.join(REPORT_DIR, f"baseline_ref_train_report.txt")
    with open(txt_path, "w") as outf:
        outf.write(f"Train: {len(train_rows)} rows\n")
        outf.write(f"Test:  {len(test_rows)} rows\n")
        outf.write(f"Intents: {len(intents)}\n\n")
        outf.write("=== CV stdout ===\n")
        outf.write(cv_proc.stdout)
        outf.write("\n=== Held-out test stdout ===\n")
        outf.write(eval_proc.stdout)
        outf.write("\n\n")
        json.dump(report, outf, indent=2)

    print(f"\nReport saved: {txt_path}")
    print(f"Model: {model_path}")
    print(f"Macro F1 (held-out): {macro.get('f1-score', 0):.4f}")

    # Parse CV F1 from stdout
    cv_macro_f1 = 0.0
    for line in cv_proc.stdout.split("\n"):
        if "macro" in line and "F1" in line:
            try:
                cv_macro_f1 = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

    held_out_f1 = macro.get("f1-score", 0)
    print(f"Macro F1 (CV):      {cv_macro_f1:.4f}")

    final_f1 = min(cv_macro_f1, held_out_f1) if cv_macro_f1 > 0 else held_out_f1
    if 0.60 <= final_f1 <= 1.0:
        print(f"\n✅ Target achieved! Macro F1 = {final_f1:.4f} (target: ~0.7)")
    else:
        print(f"\n⚠ Macro F1 = {final_f1:.4f}, below target 0.7. Consider more epochs or data augmentation.")

    # Clean up temp domain
    if os.path.exists(DOMAIN_PATH):
        os.remove(DOMAIN_PATH)

    return final_f1


if __name__ == "__main__":
    main()
