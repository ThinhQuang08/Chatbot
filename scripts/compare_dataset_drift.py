# Compare reference_normal_v3.csv vs current_trend_v3.csv using Evidently.
import sys, os, json, warnings, logging
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.db_connection import get_connection
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from data_quality_features import compute_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "text_length", "word_count", "diacritic_ratio",
    "has_emoji", "is_empty", "char_diversity", "ends_with_question",
]


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


def run():
    ref_path = os.path.join(ROOT, "data", "reference_normal_v3.csv")
    cur_path = os.path.join(ROOT, "data", "current_trend_v3.csv")

    log.info("📊 Dataset Drift Comparison")
    log.info(f"  Ref: {os.path.basename(ref_path)}")
    log.info(f"  Cur: {os.path.basename(cur_path)}")

    df_ref = pd.read_csv(ref_path)
    df_cur = pd.read_csv(cur_path)

    for df in (df_ref, df_cur):
        if "raw_text" in df.columns and "text" not in df.columns:
            df.rename(columns={"raw_text": "text"}, inplace=True)

    # --- Intent distribution analysis ---
    log.info("\n📋 Intent distribution:")
    ref_intent_pct = df_ref["intent"].value_counts(normalize=True).sort_index()
    cur_intent_pct = df_cur["intent"].value_counts(normalize=True).sort_index()
    intent_df = pd.DataFrame({"ref": ref_intent_pct, "cur": cur_intent_pct}).fillna(0)
    intent_df["delta_pp"] = (intent_df["cur"] - intent_df["ref"]) * 100
    intent_df = intent_df.sort_values("delta_pp", ascending=False)
    for intent, row in intent_df.iterrows():
        arrow = "🔴" if abs(row["delta_pp"]) > 1.0 else "🟢"
        log.info(f"    {arrow} {intent:25s} {row['ref']*100:5.1f}% → {row['cur']*100:5.1f}%  delta={row['delta_pp']:+.1f}pp")

    # --- Destination/entity analysis ---
    log.info("\n📋 Destination mentions:")
    dest_ref = df_ref["text"].str.lower().str.count("hà giang").sum()
    dest_cur = df_cur["text"].str.lower().str.count("hà giang").sum()
    log.info(f"    'hà giang' trong ref: {dest_ref} mentions")
    log.info(f"    'hà giang' trong cur: {dest_cur} mentions ({dest_cur/len(df_cur)*100:.1f}%)")

    # --- Text quality features ---
    feat_ref = compute_features(df_ref)
    feat_cur = compute_features(df_cur)
    log.info(f"\n📐 Text features computed: {len(feat_ref)} ref, {len(feat_cur)} cur")

    # --- Evidently DataDrift ---
    data_def = DataDefinition(numerical_columns=FEATURE_NAMES)
    dataset_ref = Dataset.from_pandas(feat_ref[FEATURE_NAMES], data_definition=data_def)
    dataset_cur = Dataset.from_pandas(feat_cur[FEATURE_NAMES], data_definition=data_def)

    report = Report(metrics=[DataDriftPreset(num_threshold=0.05)])
    result = report.run(reference_data=dataset_ref, current_data=dataset_cur)

    reports_dir = os.path.join(ROOT, "results", "evidently_reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(reports_dir, f"dataset_drift_comparison_{ts}.html")
    result.save_html(html_path)
    log.info(f"\n  ✅ Đã lưu Evidently report: {html_path}")

    try:
        raw = result.as_dict()
    except AttributeError:
        raw = result.dict()
    raw = make_serializable(raw)

    evidently_drift_share = 0.0
    for m in raw.get("metrics", []):
        name = m.get("metric_name", "")
        if "DriftedColumnsCount" in name:
            evidently_drift_share = m.get("value", {}).get("share", 0.0)

    # --- Custom scoring ---
    feature_results = []
    for col in FEATURE_NAMES:
        ref_vals = feat_ref[col].values
        cur_vals = feat_cur[col].values
        ref_mean = float(ref_vals.mean())
        cur_mean = float(cur_vals.mean())
        if abs(ref_mean) > 1e-9:
            delta_pct = (cur_mean - ref_mean) / abs(ref_mean) * 100
        else:
            delta_pct = (cur_mean - ref_mean) * 100

        if abs(ref_mean) > 0.01:
            penalty = abs(cur_mean - ref_mean) / abs(ref_mean)
        else:
            penalty = abs(cur_mean - ref_mean) / 0.01
        penalty = min(penalty, 1.0)

        try:
            ks_stat, ks_p = ks_2samp(ref_vals, cur_vals)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0

        feature_results.append({
            "name": col,
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "delta_pct": round(delta_pct, 2),
            "penalty": round(penalty, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
        })

    # Intent distribution penalty (KL divergence based)
    intent_ref = df_ref["intent"].value_counts(normalize=True).sort_index()
    intent_cur = df_cur["intent"].value_counts(normalize=True).sort_index()
    all_intents = sorted(set(intent_ref.index) | set(intent_cur.index))
    intent_penalty = 0.0
    for intt in all_intents:
        p_ref = intent_ref.get(intt, 0.0)
        p_cur = intent_cur.get(intt, 0.0)
        intent_penalty += abs(p_cur - p_ref) / max(len(all_intents), 1)
    intent_penalty = min(intent_penalty * len(all_intents), 1.0)

    # Destination penalty (Hà Giang skew)
    dest_penalty = min(abs(dest_cur - dest_ref) / max(len(df_cur), 1) * 10, 1.0) if dest_ref == 0 else 0

    # Combined scoring
    penalties = sorted([f["penalty"] for f in feature_results] + [intent_penalty, dest_penalty], reverse=True)
    top_k = min(3, len(penalties))
    avg_penalty = sum(penalties[:top_k]) / top_k if top_k > 0 else 0.0
    combined_score = max(0.0, 1.0 - avg_penalty)

    breached_list = [f for f in feature_results if f["penalty"] > 0.30]

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Intent distribution penalty: {intent_penalty:.2f}")
    log.info(f"  Destination (Hà Giang) penalty: {dest_penalty:.2f}")
    log.info(f"  Evidently drift share: {evidently_drift_share:.2%}")
    log.info(f"\n  Feature-level:")
    for f in feature_results:
        icon = "🔴" if f["penalty"] > 0.30 else "🟢"
        log.info(f"    {icon} {f['name']:20s} {f['ref_mean']:.3f} → {f['cur_mean']:.3f}  "
                 f"delta={f['delta_pct']:+.1f}%  penalty={f['penalty']:.2f}  KS-p={f['ks_p_value']:.4f}")
    log.info(f"\n  Combined quality score: {combined_score:.4f}")

    # Build histogram distributions
    distributions = {}
    for col in FEATURE_NAMES:
        ref_vals = feat_ref[col].values
        cur_vals = feat_cur[col].values
        all_vals = np.concatenate([ref_vals, cur_vals])
        bins = np.histogram_bin_edges(all_vals, bins=15)
        ref_hist, _ = np.histogram(ref_vals, bins=bins)
        cur_hist, _ = np.histogram(cur_vals, bins=bins)
        distributions[col] = {
            "bins": [round(float(b), 4) for b in bins],
            "ref_counts": [int(c) for c in ref_hist],
            "cur_counts": [int(c) for c in cur_hist],
        }

    # Intent distribution data
    intent_dist = {
        "labels": all_intents,
        "ref_pct": [round(intent_ref.get(i, 0) * 100, 2) for i in all_intents],
        "cur_pct": [round(intent_cur.get(i, 0) * 100, 2) for i in all_intents],
    }

    record = {
        "scenario": "reference_vs_trend",
        "ref_source": "reference_normal_v3.csv",
        "cur_source": "current_trend_v3.csv",
        "ref_rows": len(df_ref),
        "cur_rows": len(df_cur),
        "combined_score": round(combined_score, 4),
        "evidently_drift_share": round(evidently_drift_share, 4),
        "intent_penalty": round(intent_penalty, 4),
        "destination_penalty": round(dest_penalty, 4),
        "ha_giang_ref": int(dest_ref),
        "ha_giang_cur": int(dest_cur),
        "breached_count": len(breached_list),
        "total_features": len(FEATURE_NAMES) + 2,
        "features": feature_results,
        "intent_distribution": intent_dist,
        "distributions": distributions,
        "report_html_filename": f"dataset_drift_comparison_{ts}.html",
        "created_at": datetime.now().isoformat(),
    }

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("dataset_drift", json.dumps(record)),
    )
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"  ✅ Đã lưu vào mlops_reports (report_type=dataset_drift)")

    return record


if __name__ == "__main__":
    run()
