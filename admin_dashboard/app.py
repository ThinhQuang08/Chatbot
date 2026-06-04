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
from flask import Flask, jsonify, request, render_template, Response, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RASA_DIR = ROOT_DIR / "rasa_bot"
RESULTS_DIR = RASA_DIR / "results"
STATE_FILE = DATA_DIR / "review_state.json"
HISTORY_FILE = RESULTS_DIR / "training_history.json"
DRIFT_FILE = RESULTS_DIR / "model_drift_history.json"
CSV_FILE = DATA_DIR / "needs_human_review.csv"
DOMAIN_FILE = RASA_DIR / "domain.yml"
NLU_FILE = RASA_DIR / "data" / "train" / "nlu.yml"

app = Flask(__name__)


DVC_BIN = ROOT_DIR / ".dvc-venv" / "bin" / "dvc"
RASA_BIN = ROOT_DIR / ".venv" / "bin" / "rasa"


def _ensure_dvc_data():
    """Pull latest CSV data from DVC remote on startup."""
    dvc = shutil.which("dvc") or str(DVC_BIN)
    try:
        result = subprocess.run(
            [dvc, "pull"],
            capture_output=True, text=True, timeout=120
        )
        print(f"[DVC PULL] {'OK' if result.returncode == 0 else 'FAILED'}")
        if result.returncode != 0:
            print(f"[DVC PULL] stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        print("[DVC PULL] WARNING: dvc not found, skip")
    except subprocess.TimeoutExpired:
        print("[DVC PULL] WARNING: timeout (>120s), continue anyway")


ENTITY_MAP = {
    "location": [
        "hà nội", "hồ chí minh", "sài gòn", "đà nẵng", "hải phòng", "cần thơ",
        "đà lạt", "nha trang", "huế", "hội an", "vũng tàu", "phú quốc",
        "sapa", "sa pa", "hạ long", "hà giang", "cao bằng", "lai châu",
        "điện biên", "sơn la", "lào cai", "yên bái", "tuyên quang", "bắc kạn",
        "thái nguyên", "lạng sơn", "bắc giang", "phú thọ", "vĩnh phúc", "bắc ninh",
        "hải dương", "hưng yên", "thái bình", "nam định", "ninh bình", "hà nam",
        "quảng ninh", "thanh hóa", "nghệ an", "hà tĩnh", "quảng bình", "quảng trị",
        "thừa thiên huế", "quảng nam", "quảng ngãi", "bình định", "phú yên",
        "khánh hòa", "ninh thuận", "bình thuận", "kon tum", "gia lai", "đắk lắk",
        "đắk nông", "lâm đồng", "bình phước", "tây ninh", "bình dương", "đồng nai",
        "bà rịa vũng tàu", "long an", "đồng tháp", "tiền giang", "an giang",
        "bến tre", "vĩnh long", "trà vinh", "hậu giang", "kiên giang", "bạc liêu",
        "cà mau", "sóc trăng", "phan thiết", "mũi né", "phan xi păng", "fansipan",
        "côn đảo", "phú quý", "vịnh hạ long", "vịnh lan hạ", "bãi dài",
        "bãi biển", "biển mỹ khê", "biển nha trang", "biển phú quốc", "biển mũi né",
        "chùa bái đính", "tràng an", "tam cốc", "bích động", "cố đô huế",
        "phố cổ hội an", "bản cát cát", "chợ bắc hà", "chợ nổi cái răng",
        "chợ bến thành", "thung lũng tình yêu", "đồi chè cầu đất", "hồ xuân hương",
        "hồ tuyền lâm", "núi bà đen", "núi tà cú", "suối tiên", "đại nam",
        "địa đạo củ chi", "củ chi"
    ],
    "category": [
        "khách sạn", "hotel", "homestay", "resort", "nhà nghỉ", "nhà trọ",
        "hostel", "villa", "biệt thự", "khu nghỉ dưỡng", "nghỉ dưỡng",
        "chỗ ở", "chỗ nghỉ", "phòng", "căn hộ", "apartment", "farmstay",
        "glamping", "khu cắm trại", "cắm trại", "bungalow", "lodge",
        "nhà hàng", "quán ăn", "quán nhậu", "quán bar", "quán cà phê",
        "cafe", "tour", "vé", "combo", "gói du lịch",
        "biển", "núi", "miền núi", "miền biển", "miền tây", "miền bắc",
        "miền trung", "miền nam", "đồi núi", "rừng", "hồ", "sông", "suối",
        "thác", "đảo", "hòn đảo", "thành phố", "vùng quê", "đồng bằng",
        "cao nguyên", "yên tĩnh", "náo nhiệt", "sôi động", "chữa lành",
        "sang trọng", "bình dân", "giá rẻ", "tiết kiệm",
        "5 sao", "4 sao", "3 sao", "view biển", "view núi", "gần trung tâm"
    ],
    "activity": [
        "leo núi", "trekking", "cắm trại", "săn mây", "lặn san hô",
        "lặn biển", "dù lượn", "cáp treo", "đi cáp treo", "tắm biển",
        "bơi", "ngắm hoa", "chèo thuyền", "kayak", "đạp xe", "xe đạp",
        "đi bộ", "hiking", "picnic", "dã ngoại", "câu cá", "chụp ảnh",
        "sống ảo", "check in", "check-in", "tham quan", "ngắm cảnh",
        "ngắm hoàng hôn", "ngắm bình minh", "spa", "massage", "yoga",
        "thiền", "chữa lành", "healing", "mua sắm", "shopping",
        "ăn uống", "ẩm thực", "hải sản", "đặc sản", "nhậu",
        "giải trí", "vui chơi", "karaoke", "trượt tuyết", "trượt nước",
        "zipline", "đu dây", "chèo sup", "paddle", "đi tàu", "du thuyền",
        "tour đảo", "làng chài", "bản làng", "hái trái cây", "làm nông",
        "cưỡi ngựa", "chụp ảnh cưới", "tuần trăng mật", "dạo phố",
        "đi phượt", "phượt", "road trip", "camping", "bar", "pub"
    ],
    "transportation": [
        "máy bay", "vé máy bay", "xe khách", "xe đò", "tàu hỏa",
        "tàu lửa", "xe lửa", "tàu thủy", "tàu", "phà", "cano",
        "limousine", "xe limo", "xe máy", "xe gắn máy", "xe đạp điện",
        "ô tô", "ôtô", "xe hơi", "taxi", "grab", "xe ôm", "xe bus",
        "bus", "xe buýt", "xe giường nằm", "giường nằm", "xe ngồi",
        "ghế ngồi", "tàu cao tốc", "tàu cánh ngầm", "ca nô", "thuyền",
        "đi bộ"
    ]
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
        training_state["last_backup"] = str(backup_path)

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


def load_drift_history():
    if DRIFT_FILE.exists():
        with open(DRIFT_FILE, "r") as f:
            return json.load(f)
    return []


def save_drift_history(history):
    DRIFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def append_log_to_file(log_line):
    log_path = RESULTS_DIR / "training_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)


def run_cv_and_get_metrics(data_path, out_dir, log_fn):
    """Run rasa test nlu --cross-validation and return metrics dict."""
    rasa_bin = shutil.which("rasa") or str(RASA_BIN)
    eval_proc = subprocess.run(
        [rasa_bin, "test", "nlu", "--cross-validation", "--folds", "3",
         "--nlu", str(data_path), "--out", str(out_dir)],
        cwd=str(RASA_DIR),
        capture_output=True, text=True, timeout=3600
    )

    if eval_proc.stdout.strip():
        log_fn(eval_proc.stdout.strip())
    if eval_proc.stderr.strip():
        log_fn("STDERR: " + eval_proc.stderr.strip())

    report_path = out_dir / "intent_report.json"
    if report_path.exists():
        with open(report_path, "r") as f:
            report = json.load(f)
        macro_avg = report.get("macro avg", {})
        return {
            "f1_score": macro_avg.get("f1-score", 0.0),
            "accuracy": report.get("accuracy", 0.0),
            "precision": macro_avg.get("precision", 0.0),
            "recall": macro_avg.get("recall", 0.0)
        }
    return None


def run_training():
    global training_state
    training_state["running"] = True
    training_state["logs"] = ""
    training_state["start_time"] = datetime.now().isoformat()
    training_state["error"] = None
    training_state["metrics"] = None

    # Capture backup path before thread starts (avoid race with export)
    last_backup = training_state.get("last_backup")

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        training_state["logs"] += line + "\n"
        append_log_to_file(line + "\n")
        print(line)

    try:
        start_time = time.time()
        rasa_bin = shutil.which("rasa") or str(RASA_BIN)

        # ---- Step 0: Pre-drift evaluation on OLD data (before adding new) ----
        new_examples_count = 0
        if last_backup and Path(last_backup).exists():
            log("🧪 Đánh giá drift trước khi train (pre-test trên dữ liệu cũ)...")
            tmp_dir = RASA_DIR / "data" / "tmp_drift_eval"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(last_backup, tmp_dir / "nlu.yml")

            out_dir = RESULTS_DIR / "eval_pre"
            pre_metrics = run_cv_and_get_metrics(tmp_dir, out_dir, log)

            shutil.rmtree(tmp_dir, ignore_errors=True)

            if pre_metrics:
                drift_history = load_drift_history()
                drift_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": "pre",
                    **pre_metrics,
                    "new_examples_count": None
                })
                save_drift_history(drift_history)
                log(f"📊 Pre-test — F1: {pre_metrics['f1_score']:.4f}, "
                    f"Accuracy: {pre_metrics['accuracy']:.4f}")
            else:
                log("⚠️ Pre-test: Không tìm thấy intent_report.json, bỏ qua.")
        else:
            log("ℹ️ Không có backup dữ liệu cũ, bỏ qua pre-test drift.")

        # ---- Step 1: Train ----
        log("🚀 Bắt đầu huấn luyện Rasa model...")
        train_proc = subprocess.run(
            [rasa_bin, "train", "--data", "data/train"],
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

        # ---- Step 2: Post-training evaluation ----
        log("🧪 Đang đánh giá model với cross-validation (3 folds)...")
        eval_dir = RESULTS_DIR / "eval_post"
        post_metrics = run_cv_and_get_metrics(
            RASA_DIR / "data" / "train", eval_dir, log
        )

        if post_metrics:
            training_state["metrics"] = post_metrics
            training_duration = round(time.time() - start_time, 2)

            # Save to existing training_history (backward compatible)
            history = load_training_history()
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": training_duration,
                **post_metrics
            })
            save_training_history(history)

            # Save to drift_history as "post"
            drift_history = load_drift_history()
            drift_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "phase": "post",
                **post_metrics,
                "new_examples_count": new_examples_count,
                "duration_seconds": training_duration
            })
            save_drift_history(drift_history)

            log(f"📊 F1 Score:  {post_metrics['f1_score']:.4f}")
            log(f"📊 Accuracy: {post_metrics['accuracy']:.4f}")
            log("✅ Đánh giá hoàn tất!")
        else:
            log("⚠️ Không tìm thấy intent_report.json. Training có thể chưa đủ data test.")

        # Clear backup reference after successful cycle
        training_state["last_backup"] = None

    except subprocess.TimeoutExpired:
        training_state["error"] = "Training timeout (>30 phút)"
        log("❌ Training timeout (>30 phút)")
    except Exception as e:
        training_state["error"] = str(e)
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


@app.route("/api/model-metrics")
def model_metrics():
    history = load_drift_history()
    if not history:
        return jsonify({"labels": [], "datasets": [], "latest": None})

    labels = [e["timestamp"] for e in history]

    def series(key):
        return [e.get(key, 0.0) for e in history]

    datasets = [
        {"label": "F1 Score",  "data": series("f1_score"),
         "borderColor": "#6c5ce7", "backgroundColor": "rgba(108,92,231,0.1)"},
        {"label": "Accuracy",  "data": series("accuracy"),
         "borderColor": "#00b894", "backgroundColor": "rgba(0,184,148,0.1)"},
        {"label": "Precision", "data": series("precision"),
         "borderColor": "#fdcb6e", "backgroundColor": "rgba(253,203,110,0.1)"},
        {"label": "Recall",    "data": series("recall"),
         "borderColor": "#e17055", "backgroundColor": "rgba(225,112,85,0.1)"},
    ]

    latest = dict(history[-1]) if history else None
    if latest and len(history) >= 2:
        prev = history[-2]
        latest["deltas"] = {
            k: round(latest[k] - prev[k], 4)
            for k in ["f1_score", "accuracy", "precision", "recall"]
        }

    return jsonify({"labels": labels, "datasets": datasets, "latest": latest})


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


@app.route("/api/results-images")
def list_results_images():
    """Return list of .png images available in results/ directory."""
    img_dir = RESULTS_DIR
    valid_prefixes = ("intent_", "DIETClassifier_", "RegexEntityExtractor_")
    valid_suffixes = ("confusion_matrix.png", "histogram.png")

    images = []
    if img_dir.exists():
        for f in sorted(img_dir.iterdir()):
            if f.suffix == ".png" and f.name.startswith(valid_prefixes):
                images.append({
                    "filename": f.name,
                    "label": f.name.replace("_", " ").replace(".png", "").title(),
                    "url": f"/results-img/{f.name}"
                })
    return jsonify(images)


@app.route("/results-img/<filename>")
def serve_results_image(filename):
    """Serve a .png image from the results/ directory."""
    return send_from_directory(str(RESULTS_DIR), filename)


if __name__ == "__main__":
    _ensure_dvc_data()
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
