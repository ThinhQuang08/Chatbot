import sys, os, json, warnings, smtplib, ssl, logging
from email.message import EmailMessage
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from database.db_connection import get_connection
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from config.settings import SMTP_EMAIL, SMTP_PASSWORD, SMTP_HOST, SMTP_PORT

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.50"))
FEATURE_NAMES = [
    "text_length", "word_count", "diacritic_ratio",
    "has_emoji", "is_empty", "char_diversity", "ends_with_question",
]

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from data_quality_features import compute_features  # noqa


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "value"):
        return obj.value
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def send_alert_email(score, threshold, breached, feature_details, cur_rows, ref_rows):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        log.info("  [EMAIL] SMTP chưa config — bỏ qua gửi mail")
        body_lines = [f"Quality Score: {score:.2f} (threshold: {threshold})"]
        body_lines.append(f"Breached: {len(breached)}/{len(FEATURE_NAMES)}")
        for f in breached:
            body_lines.append(
                f"  🔴 {f['name']}: {f['ref_mean']} → {f['cur_mean']} "
                f"(delta={f['delta_pct']:+.1f}%, penalty={f['penalty']:.2f})"
            )
        log.info(f"  [EMAIL] Nội dung:\n" + "\n".join(body_lines))
        return False

    subject = f"[CẢNH BÁO] Data Quality Score = {score:.2f} (dưới ngưỡng {threshold})"
    body = f"""
Data Quality Monitor - Cảnh báo chất lượng dữ liệu

Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Quality Score: {score:.2f}  (ngưỡng: {threshold})
Dòng tham chiếu: {ref_rows}
Dòng hiện tại: {cur_rows}

Features bị drift ({len(breached)}/{len(FEATURE_NAMES)}):
"""
    for f in breached:
        body += f"  🔴 {f['name']}: {f['ref_mean']} → {f['cur_mean']}  (delta={f['delta_pct']:+.1f}%)\n"

    body += "\nFeatures OK:\n"
    for f in feature_details:
        if f["penalty"] <= 0.30:
            body += f"  🟢 {f['name']}: {f['ref_mean']} → {f['cur_mean']}\n"

    body += "\nVui lòng kiểm tra dashboard để biết thêm chi tiết."
    body = body.strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = SMTP_EMAIL
    msg.set_content(body)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=ctx)
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        log.info(f"  [EMAIL] Đã gửi cảnh báo tới {SMTP_EMAIL}")
        return True
    except Exception as e:
        log.warning(f"  [EMAIL] Gửi thất bại: {e}")
        return False


def run(ref_path, cur_path):
    log.info("📊 Data Quality Monitor")
    log.info(f"  Ref: {ref_path}")
    log.info(f"  Cur: {cur_path}")

    df_ref = pd.read_csv(ref_path)
    df_cur = pd.read_csv(cur_path)

    text_col = "text" if "text" in df_ref.columns else df_ref.columns[0]
    df_ref = df_ref[~df_ref[text_col].isna()]
    df_cur = df_cur[~df_cur[text_col].isna()]

    feat_ref = compute_features(df_ref)
    feat_cur = compute_features(df_cur)
    log.info(f"  Ref features: {len(feat_ref)} rows")
    log.info(f"  Cur features: {len(feat_cur)} rows")

    data_def = DataDefinition(
        numerical_columns=FEATURE_NAMES,
    )
    dataset_ref = Dataset.from_pandas(feat_ref[FEATURE_NAMES], data_definition=data_def)
    dataset_cur = Dataset.from_pandas(feat_cur[FEATURE_NAMES], data_definition=data_def)

    report = Report(metrics=[DataDriftPreset(num_threshold=0.05)])
    result = report.run(reference_data=dataset_ref, current_data=dataset_cur)

    # Save interactive HTML report
    reports_dir = os.path.join(ROOT, "results", "evidently_reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(reports_dir, f"data_quality_drift_{ts}.html")
    result.save_html(html_path)
    log.info(f"  ✅ Đã lưu Evidently report: {html_path}")

    try:
        raw = result.as_dict()
    except AttributeError:
        raw = result.dict()
    raw = make_serializable(raw)

    drift_share = 0.0
    for m in raw.get("metrics", []):
        name = m.get("metric_name", "")
        if "DriftedColumnsCount" in name:
            drift_share = m.get("value", {}).get("share", 0.0)

    # Custom quality scoring — drift from Evidently's KS test is unreliable for
    # binary features (is_empty, has_emoji). Use relative delta scoring instead.
    feature_results = []
    for col in FEATURE_NAMES:
        ref_mean = float(feat_ref[col].mean())
        cur_mean = float(feat_cur[col].mean())
        if abs(ref_mean) > 1e-9:
            delta_pct = (cur_mean - ref_mean) / abs(ref_mean) * 100
        else:
            delta_pct = (cur_mean - ref_mean) * 100

        if abs(ref_mean) > 0.01:
            penalty = abs(cur_mean - ref_mean) / abs(ref_mean)
        else:
            penalty = abs(cur_mean - ref_mean) / 0.01
        penalty = min(penalty, 1.0)

        feature_results.append({
            "name": col,
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "delta_pct": round(delta_pct, 2),
            "penalty": round(penalty, 4),
        })

    total = len(FEATURE_NAMES)
    penalties = sorted([f["penalty"] for f in feature_results], reverse=True)
    # Use top-3 average for sensitivity — one bad feature shouldn't dominate
    # but multiple bad features should drop score significantly
    top_k = min(3, len(penalties))
    avg_penalty = sum(penalties[:top_k]) / top_k if top_k > 0 else 0.0
    quality_score = max(0.0, 1.0 - avg_penalty)

    breached_list = [f for f in feature_results if f["penalty"] > 0.30]

    log.info(f"  Quality Score: {quality_score:.2f}  [threshold: {QUALITY_THRESHOLD}]")
    log.info(f"  Breached: {len(breached_list)}/{total}")

    for f in feature_results:
        icon = "🔴" if f["penalty"] > 0.30 else "🟢"
        log.info(f"    {icon} {f['name']:20s} {f['ref_mean']:.3f} → {f['cur_mean']:.3f}  "
                 f"delta={f['delta_pct']:+.1f}%  penalty={f['penalty']:.2f}")

    record = {
        "quality_score": round(quality_score, 4),
        "threshold": QUALITY_THRESHOLD,
        "drift_share": round(drift_share, 4),
        "ref_rows": len(feat_ref),
        "cur_rows": len(feat_cur),
        "breached_count": len(breached_list),
        "total_features": total,
        "features": feature_results,
        "ref_source": os.path.basename(ref_path),
        "cur_source": os.path.basename(cur_path),
        "email_sent": False,
        "report_html_filename": f"data_quality_drift_{ts}.html",
    }

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("data_quality_drift", json.dumps(record)),
    )
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"  ✅ Đã lưu vào mlops_reports")

    if quality_score < QUALITY_THRESHOLD:
        log.info(f"  Quality Score {quality_score:.2f} < {QUALITY_THRESHOLD} → gửi email...")
        sent = send_alert_email(
            score=quality_score,
            threshold=QUALITY_THRESHOLD,
            breached=breached_list,
            feature_details=feature_results,
            cur_rows=len(feat_cur),
            ref_rows=len(feat_ref),
        )
        if sent:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE mlops_reports SET metrics = jsonb_set(metrics, '{email_sent}', 'true'::jsonb) "
                "WHERE id = (SELECT max(id) FROM mlops_reports WHERE report_type = 'data_quality_drift')"
            )
            conn.commit()
            cur.close()
            conn.close()
    else:
        log.info(f"  ✅ Quality Score {quality_score:.2f} >= {QUALITY_THRESHOLD} (OK)")

    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to reference CSV")
    parser.add_argument("--cur", required=True, help="Path to current CSV")
    args = parser.parse_args()

    run(ref_path=args.ref, cur_path=args.cur)
