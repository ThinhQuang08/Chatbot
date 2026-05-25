import os
import sys
import json
import time
import shutil
import threading
import subprocess
import re
import csv
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template, Response

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RASA_DIR = ROOT_DIR / "rasa_bot"
RESULTS_DIR = RASA_DIR / "results"
STATE_FILE = DATA_DIR / "review_state.json"
HISTORY_FILE = RESULTS_DIR / "training_history.json"
CSV_FILE = DATA_DIR / "needs_human_review.csv"
DOMAIN_FILE = RASA_DIR / "domain.yml"
NLU_FILE = RASA_DIR / "data" / "train" / "nlu.yml"

app = Flask(__name__)

ENTITY_MAP = {
    "destination": ["đà lạt", "nha trang", "phú quốc", "sapa", "sa pa", "đà nẵng", "hà nội", "sài gòn", "vũng tàu", "hội an"],
    "category": ["khách sạn", "resort", "homestay", "nhà nghỉ", "nhà hàng", "nghỉ dưỡng", "villa", "tour", "chỗ ở"],
    "activity": ["cắm trại", "trekking", "leo núi", "đạp xe", "xe đạp", "dù lượn", "lặn", "check in", "tour đảo", "hoạt động vui chơi"]
}

training_state = {
    "running": False,
    "logs": "",
    "start_time": None,
    "end_time": None,
    "metrics": None,
    "error": None
}


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_csv_data():
    if not CSV_FILE.exists():
        return []
    df = pd.read_csv(CSV_FILE)
    df = df.where(pd.notna(df), None)
    return df.to_dict("records")


def load_intents():
    intents = []
    in_intents = False
    with open(DOMAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "intents:":
                in_intents = True
                continue
            if in_intents:
                if stripped.startswith("- "):
                    intent = stripped[2:].strip()
                    if intent and not intent.startswith("#"):
                        intents.append(intent)
                elif stripped and not (stripped.startswith("#") or stripped == ""):
                    break
    return intents


def auto_annotate_entities(text):
    annotated_text = str(text)
    for entity_type, keywords in ENTITY_MAP.items():
        keywords_sorted = sorted(keywords, key=len, reverse=True)
        for kw in keywords_sorted:
            pattern = re.compile(rf'(?<!\[)\b({re.escape(kw)})\b(?!\])', re.IGNORECASE)
            annotated_text = pattern.sub(rf'[\1]({entity_type})', annotated_text)
    return annotated_text


def export_to_nlu(approved_rows):
    backup_name = None
    if NLU_FILE.exists():
        backup_name = f"nlu_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
        backup_path = RASA_DIR / "data" / "backups" / backup_name
        shutil.copy(NLU_FILE, backup_path)

    intent_groups = {}
    for row in approved_rows:
        intent = row["intent"]
        text = row.get("corrected_text") or row.get("cleaned_text", "")
        text_natural = text.replace("_", " ")
        text_annotated = auto_annotate_entities(text_natural)

        if intent not in intent_groups:
            intent_groups[intent] = set()
        intent_groups[intent].add(text_annotated)

    with open(NLU_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n# --- HUMAN REVIEWED DATA ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        for intent, examples in intent_groups.items():
            f.write(f"  - intent: {intent}\n")
            f.write(f"    examples: |\n")
            for text in sorted(examples):
                f.write(f"      - {text}\n")

    return {
        "backup_file": backup_name,
        "intents_added": list(intent_groups.keys()),
        "total_examples": sum(len(v) for v in intent_groups.values())
    }


def remove_reviewed_rows_from_csv():
    if not CSV_FILE.exists():
        return
    state = load_state()
    if not state:
        return

    df = pd.read_csv(CSV_FILE)
    ids_to_remove = set(state.keys())
    df_filtered = df[~df['id'].astype(str).isin(ids_to_remove)]

    removed_count = len(df) - len(df_filtered)
    if removed_count > 0:
        df_filtered.to_csv(CSV_FILE, index=False)

    save_state({})
    return removed_count


def load_training_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_training_history(history):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_log_to_file(log_line):
    log_path = RESULTS_DIR / "training_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)


def run_training():
    global training_state
    training_state["running"] = True
    training_state["logs"] = ""
    training_state["start_time"] = datetime.now().isoformat()
    training_state["error"] = None
    training_state["metrics"] = None

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        training_state["logs"] += line + "\n"
        append_log_to_file(line + "\n")
        print(line)

    try:
        start_time = time.time()

        log("🚀 Bắt đầu huấn luyện Rasa model...")
        train_proc = subprocess.run(
            ["rasa", "train", "--data", "data/train"],
            cwd=str(RASA_DIR),
            capture_output=True, text=True, timeout=1800
        )

        if train_proc.stdout.strip():
            log(train_proc.stdout.strip())
        if train_proc.stderr.strip():
            log("STDERR: " + train_proc.stderr.strip())

        if train_proc.returncode != 0:
            training_state["error"] = "Training failed"
            log("❌ Training thất bại!")
            training_state["end_time"] = datetime.now().isoformat()
            training_state["running"] = False
            return

        log("✅ Training thành công!")

        log("🧪 Đang đánh giá model với cross-validation (3 folds)...")
        eval_proc = subprocess.run(
            ["rasa", "test", "nlu", "--cross-validation", "--folds", "3", "--data", "data/train"],
            cwd=str(RASA_DIR),
            capture_output=True, text=True, timeout=3600
        )

        if eval_proc.stdout.strip():
            log(eval_proc.stdout.strip())
        if eval_proc.stderr.strip():
            log("STDERR: " + eval_proc.stderr.strip())

        report_path = RESULTS_DIR / "intent_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                report = json.load(f)

            macro_avg = report.get("macro avg", {})
            metrics = {
                "f1_score": macro_avg.get("f1-score", 0.0),
                "accuracy": report.get("accuracy", 0.0),
                "precision": macro_avg.get("precision", 0.0),
                "recall": macro_avg.get("recall", 0.0)
            }
            training_state["metrics"] = metrics
            training_duration = round(time.time() - start_time, 2)

            history = load_training_history()
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": training_duration,
                **metrics
            })
            save_training_history(history)

            log(f"📊 F1 Score:  {metrics['f1_score']:.4f}")
            log(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            log("✅ Đánh giá hoàn tất!")
        else:
            log("⚠️ Không tìm thấy intent_report.json. Training có thể chưa đủ data test.")

    except subprocess.TimeoutExpired:
        training_state["error"] = "Training timeout (>30 phút)"
        log("❌ Training timeout (>30 phút)")
    except Exception as e:
        training_state["error"] = str(e)
        log(f"❌ Lỗi: {e}")
    finally:
        training_state["end_time"] = datetime.now().isoformat()
        training_state["running"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    rows = load_csv_data()
    state = load_state()

    result = []
    for row in rows:
        row_id = str(row.get("id", ""))
        item = {
            "id": row_id,
            "session_id": row.get("session_id", ""),
            "raw_text": row.get("raw_text", ""),
            "cleaned_text": row.get("cleaned_text", ""),
            "snorkel_intent": row.get("snorkel_intent", ""),
            "snorkel_confidence": row.get("snorkel_confidence", ""),
            "status": "pending"
        }
        if row_id in state:
            s = state[row_id]
            item["status"] = s.get("status", "pending")
            if "intent" in s:
                item["labeled_intent"] = s["intent"]
            if "corrected_text" in s and s.get("corrected_text"):
                item["corrected_text"] = s["corrected_text"]

        result.append(item)

    total = len(result)
    pending = sum(1 for r in result if r["status"] == "pending")
    approved = sum(1 for r in result if r["status"] == "approved")
    rejected = sum(1 for r in result if r["status"] == "rejected")

    return jsonify({
        "rows": result,
        "stats": {"total": total, "pending": pending, "approved": approved, "rejected": rejected}
    })


@app.route("/api/intents")
def get_intents():
    return jsonify(load_intents())


@app.route("/api/label", methods=["POST"])
def label_row():
    data = request.json
    row_id = str(data.get("id"))
    intent = data.get("intent")
    corrected_text = data.get("corrected_text", "")

    state = load_state()
    state[row_id] = {
        "status": "approved",
        "intent": intent,
        "corrected_text": corrected_text
    }
    save_state(state)
    return jsonify({"success": True})


@app.route("/api/reject", methods=["POST"])
def reject_row():
    data = request.json
    row_id = str(data.get("id"))

    state = load_state()
    state[row_id] = {"status": "rejected"}
    save_state(state)
    return jsonify({"success": True})


@app.route("/api/batch", methods=["POST"])
def batch_action():
    data = request.json
    ids = [str(i) for i in data.get("ids", [])]
    action = data.get("action")

    state = load_state()
    for row_id in ids:
        if action == "reject":
            state[row_id] = {"status": "rejected"}
        elif action == "label":
            intent = data.get("intent")
            state[row_id] = {
                "status": "approved",
                "intent": intent,
                "corrected_text": ""
            }
    save_state(state)
    return jsonify({"success": True, "processed": len(ids)})


@app.route("/api/export-nlu", methods=["POST"])
def export_nlu():
    state = load_state()
    rows = load_csv_data()

    approved = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in state and state[row_id].get("status") == "approved":
            approved.append({
                "intent": state[row_id].get("intent", ""),
                "corrected_text": state[row_id].get("corrected_text", ""),
                "cleaned_text": row.get("cleaned_text", ""),
                "raw_text": row.get("raw_text", "")
            })

    if not approved:
        return jsonify({"success": False, "error": "No approved rows to export"})

    result = export_to_nlu(approved)
    removed = remove_reviewed_rows_from_csv()
    result["removed_from_csv"] = removed
    return jsonify({"success": True, **result})


@app.route("/api/retrain", methods=["POST"])
def retrain():
    if training_state["running"]:
        return jsonify({"success": False, "error": "Training already in progress"})

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({"success": True})


@app.route("/api/train-status")
def train_status():
    return jsonify(training_state)


@app.route("/api/train-history")
def train_history():
    return jsonify(load_training_history())


@app.route("/api/reset", methods=["POST"])
def reset_review():
    data = request.json
    row_id = str(data.get("id"))
    state = load_state()
    if row_id in state:
        del state[row_id]
        save_state(state)
    return jsonify({"success": True})


@app.route("/api/intent-distribution")
def intent_distribution():
    rows = load_csv_data()
    counts = {}
    for row in rows:
        intent = str(row.get("snorkel_intent", "")).strip()
        if not intent or intent == "nan":
            intent = "UNKNOWN"
        counts[intent] = counts.get(intent, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    return jsonify({
        "labels": [item[0] for item in sorted_items],
        "counts": [item[1] for item in sorted_items],
        "total": sum(item[1] for item in sorted_items)
    })


@app.route("/api/export-csv", methods=["POST"])
def export_csv():
    state = load_state()
    rows = load_csv_data()

    approved = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in state and state[row_id].get("status") == "approved":
            s = state[row_id]
            approved.append({
                "id": row_id,
                "session_id": row.get("session_id", ""),
                "raw_text": row.get("raw_text", ""),
                "cleaned_text": row.get("cleaned_text", ""),
                "labeled_intent": s.get("intent", ""),
                "corrected_text": s.get("corrected_text", ""),
                "snorkel_intent": row.get("snorkel_intent", ""),
                "snorkel_confidence": row.get("snorkel_confidence", ""),
                "status": "approved"
            })

    if not approved:
        return jsonify({"success": False, "error": "No approved rows to export"})

    remove_reviewed_rows_from_csv()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "session_id", "raw_text", "cleaned_text",
                     "labeled_intent", "corrected_text", "snorkel_intent",
                     "snorkel_confidence", "status"])
    for row in approved:
        writer.writerow([
            row["id"], row["session_id"], row["raw_text"], row["cleaned_text"],
            row["labeled_intent"], row["corrected_text"], row["snorkel_intent"],
            row["snorkel_confidence"], row["status"]
        ])

    csv_content = output.getvalue()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reviewed_data_{timestamp}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-disposition": f"attachment; filename={filename}",
            "Content-type": "text/csv; charset=utf-8"
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
