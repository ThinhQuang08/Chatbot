"""
Demo pipeline: Text Drift -> F1 Degradation -> Recovery
Usage: conda run -n monitor-env python scripts/demo_drift_pipeline.py
"""
import sys, os, json, csv, shutil, subprocess, random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
RASA_DIR = BASE / "rasa_bot"
DATA_DIR = BASE / "data"
SCRIPTS_DIR = BASE / "scripts"
DEMO_DIR = RASA_DIR / "data" / "demo"
OUT_DIR = RASA_DIR / "results" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───
TEST_SIZE = 0.1   # 10% test, 10% val, 80% train
VAL_SIZE = 0.1
REF_CSV = DATA_DIR / "reference_normal_v3.csv"
CUR_CSV = DATA_DIR / "current_trend_v3.csv"
ORIG_NLU = RASA_DIR / "data" / "train" / "nlu.yml"
RASA_BIN = shutil.which("rasa") or str(RASA_DIR.parent / ".venv" / "bin" / "rasa")
RESULTS_LOG = []


def log(msg):
    timestamp = __import__("datetime").datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    RESULTS_LOG.append(line)
    print(line)


def csv_to_nlu(csv_path, intents_allowlist=None):
    """Convert CSV to nlu.yml format string."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            intent = row.get("intent", "").strip()
            if text and intent:
                if intents_allowlist and intent not in intents_allowlist:
                    continue
                rows.append((text, intent))

    groups = {}
    for text, intent in rows:
        groups.setdefault(intent, []).append(text)

    lines = ['version: "3.1"', "", "nlu:"]
    for intent in sorted(groups.keys()):
        lines.append(f"  - intent: {intent}")
        lines.append('    examples: |')
        for text in groups[intent]:
            lines.append(f"      - {text}")
        lines.append("")
    return "\n".join(lines)


def split_data(rows, train_ratio=0.8, val_ratio=0.1):
    """Split rows into train/val/test, stratified by intent."""
    by_intent = {}
    for text, intent in rows:
        by_intent.setdefault(intent, []).append((text, intent))

    train, val, test = [], [], []
    for intent, items in by_intent.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * TEST_SIZE))
        n_val = max(1, round(n * VAL_SIZE))
        n_train = n - n_test - n_val
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_nlu_file(path, rows):
    """Write list of (text, intent) to nlu.yml file."""
    groups = {}
    for text, intent in rows:
        groups.setdefault(intent, []).append(text)

    lines = ['version: "3.1"', "", "nlu:"]
    for intent in sorted(groups.keys()):
        lines.append(f"  - intent: {intent}")
        lines.append('    examples: |')
        for text in groups[intent]:
            lines.append(f"      - {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_rasa_train(data_path, model_name, extra_args=None):
    """Train a Rasa model and return the model path."""
    log(f"Training model '{model_name}'...")
    cmd = [RASA_BIN, "train", "--data", str(data_path), "--config", str(RASA_DIR / "config.yml"),
           "--out", str(RASA_DIR / "models"), "--fixed-model-name", model_name]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=str(RASA_DIR), capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        log(f"Train FAILED: {proc.stderr[-500:]}")
        return None
    model_path = RASA_DIR / "models" / f"{model_name}.tar.gz"
    if model_path.exists():
        log(f"Model saved: {model_path}")
        return model_path
    log(f"Model file not found at {model_path}")
    return None


def run_rasa_test(model_path, test_nlu_file, out_subdir):
    """Run rasa test nlu and return metrics dict."""
    out_dir = OUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Testing model on {test_nlu_file.name}...")
    cmd = [RASA_BIN, "test", "nlu", "--model", str(model_path),
           "--nlu", str(test_nlu_file), "--out", str(out_dir)]
    proc = subprocess.run(cmd, cwd=str(RASA_DIR), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        log(f"Test FAILED: {proc.stderr[-500:]}")
        return None
    report_path = out_dir / "intent_report.json"
    if not report_path.exists():
        log(f"No intent_report.json found in {out_dir}")
        return None
    with open(report_path) as f:
        report = json.load(f)
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    metrics = {
        "f1_score": macro.get("f1-score", 0),
        "precision": macro.get("precision", 0),
        "recall": macro.get("recall", 0),
        "accuracy": report.get("accuracy", 0),
        "weighted_f1": weighted.get("f1-score", 0),
    }
    log(f"  F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    return metrics


def print_summary_table(results):
    """Print summary table of all evaluation results."""
    print("\n" + "=" * 72)
    print("  DEMO SUMMARY: TEXT DRIFT -> F1 DEGRADATION -> RECOVERY")
    print("=" * 72)
    print(f"  {'Phase':<25s} {'F1':<10s} {'Precision':<12s} {'Recall':<10s} {'Accuracy':<10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
    for phase, m in results:
        if m:
            print(f"  {phase:<25s} {m['f1_score']:<10.4f} {m['precision']:<12.4f} {m['recall']:<10.4f} {m['accuracy']:<10.4f}")
        else:
            print(f"  {phase:<25s} {'FAILED':<10s}")
    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    log("=== DEMO: Text Drift -> F1 Degradation -> Recovery ===")
    print()

    # ─── Step 0: Prepare directories & read CSVs ───
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # Read CSVs
    ref_rows = []
    with open(REF_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref_rows.append((row["text"].strip(), row["intent"].strip()))
    cur_rows = []
    with open(CUR_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cur_rows.append((row["text"].strip(), row["intent"].strip()))

    log(f"Loaded {len(ref_rows)} reference rows, {len(cur_rows)} current trend rows")

    # ─── Step 1: Split reference -> train/val/test ───
    log("Splitting reference data 80:10:10...")
    ref_train, ref_val, ref_test = split_data(ref_rows)
    log(f"  Train: {len(ref_train)}, Val: {len(ref_val)}, Test: {len(ref_test)}")

    # Get intents in reference for filtering
    ref_intents = set(i for _, i in ref_rows)

    # Write split files as nlu.yml
    write_nlu_file(DEMO_DIR / "ref_train.yml", ref_train)
    write_nlu_file(DEMO_DIR / "ref_val.yml", ref_val)
    write_nlu_file(DEMO_DIR / "ref_test.yml", ref_test)

    # Current trend as nlu.yml
    cur_intents = set(i for _, i in cur_rows)
    write_nlu_file(DEMO_DIR / "cur_trend.yml", cur_rows)

    # ─── Step 2: Prepare baseline training data ───
    # Copy original nlu.yml + ref_train + ref_val
    log("Preparing baseline training data...")
    baseline_data_dir = DEMO_DIR / "baseline_data"
    baseline_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy original nlu.yml (only intents that exist in reference, plus essential intents)
    # We keep ALL original intents but combine with ref data
    shutil.copy(ORIG_NLU, baseline_data_dir / "nlu_original.yml")
    shutil.copy(DEMO_DIR / "ref_train.yml", baseline_data_dir / "ref_train.yml")
    shutil.copy(DEMO_DIR / "ref_val.yml", baseline_data_dir / "ref_val.yml")

    # ─── Step 3: Train baseline model ───
    log("\n" + "=" * 60)
    log("PHASE 1: TRAINING BASELINE MODEL")
    log("=" * 60)
    model_baseline = run_rasa_train(baseline_data_dir, "demo_baseline")
    metrics_recovery = None
    metrics_recovery_ref = None

    if not model_baseline:
        log("BASELINE TRAINING FAILED. Aborting.")
        return

    # ─── Step 4: Evaluate baseline on reference test → BASELINE metrics ───
    log("\n" + "=" * 60)
    log("PHASE 2: EVALUATE BASELINE ON REFERENCE TEST")
    log("=" * 60)
    metrics_baseline = run_rasa_test(model_baseline, DEMO_DIR / "ref_test.yml", "baseline_ref_test")

    # ─── Step 5: Evaluate baseline on current trend → DRIFT metrics ───
    log("\n" + "=" * 60)
    log("PHASE 3: EVALUATE BASELINE ON CURRENT TREND (70% HA GIANG)")
    log("=" * 60)
    metrics_drift = run_rasa_test(model_baseline, DEMO_DIR / "cur_trend.yml", "baseline_cur_trend")

    # ─── Step 6: Recovery - augment training with current trend data ───
    log("\n" + "=" * 60)
    log("PHASE 4: RECOVERY - RETRAIN WITH AUGMENTED DATA")
    log("=" * 60)

    log("Augmenting training data with current trend (hot trend)...")
    recovery_data_dir = DEMO_DIR / "recovery_data"
    recovery_data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(ORIG_NLU, recovery_data_dir / "nlu_original.yml")
    shutil.copy(DEMO_DIR / "ref_train.yml", recovery_data_dir / "ref_train.yml")
    shutil.copy(DEMO_DIR / "ref_val.yml", recovery_data_dir / "ref_val.yml")

    # We augment with a sample of the trend data (half of it for realistic recovery)
    augmented_cur = random.sample(cur_rows, min(1500, len(cur_rows)))
    write_nlu_file(recovery_data_dir / "cur_trend_augment.yml", augmented_cur)

    # ─── Step 7: Retrain with augmented data ───
    model_recovery = run_rasa_train(recovery_data_dir, "demo_recovery")

    if not model_recovery:
        log("RECOVERY TRAINING FAILED.")
    else:
        # ─── Step 8: Evaluate recovery model on current trend → RECOVERY metrics ───
        log("\n" + "=" * 60)
        log("PHASE 5: EVALUATE RECOVERY MODEL ON CURRENT TREND")
        log("=" * 60)
        metrics_recovery = run_rasa_test(model_recovery, DEMO_DIR / "cur_trend.yml", "recovery_cur_trend")

        # ─── Step 9: Evaluate recovery model on reference test (regression check) ───
        log("\n" + "=" * 60)
        log("PHASE 6: REGRESSION CHECK - RECOVERY ON REFERENCE TEST")
        log("=" * 60)
        metrics_recovery_ref = run_rasa_test(model_recovery, DEMO_DIR / "ref_test.yml", "recovery_ref_test")

    # ─── Summary ───
    results = [
        ("1-Baseline (ref test)", metrics_baseline),
        ("2-Drift (trend test)", metrics_drift),
        ("3-Recovery (trend test)", metrics_recovery if model_recovery else None),
        ("4-Regression (ref test)", metrics_recovery_ref if model_recovery else None),
    ]
    print_summary_table(results)

    # Save results to JSON
    summary = {"phases": [], "config": {"ref_train": len(ref_train), "ref_val": len(ref_val),
                                        "ref_test": len(ref_test), "cur_trend": len(cur_rows)}}
    for phase, m in results:
        entry = {"phase": phase}
        if m:
            entry.update(m)
        else:
            entry["error"] = "FAILED"
        summary["phases"].append(entry)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Summary saved to {OUT_DIR / 'summary.json'}")

    print("\n".join(RESULTS_LOG))
    print(f"\nFull log saved to {OUT_DIR / 'pipeline_log.txt'}")
    (OUT_DIR / "pipeline_log.txt").write_text("\n".join(RESULTS_LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
