# Detect intent + entity drift for hot trend scenario.
import sys, os, json, warnings, logging, re
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon

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

QUALITY_THRESHOLD = float(os.environ.get("INTENT_DRIFT_THRESHOLD", "0.50"))

VIETNAM_DESTINATIONS = [
    "hà nội", "hồ chí minh", "sài gòn", "đà nẵng", "hải phòng", "cần thơ",
    "đà lạt", "nha trang", "huế", "hội an", "vũng tàu", "phú quốc",
    "sapa", "sa pa", "hạ long", "hà giang", "cao bằng", "lai châu",
    "điện biên", "sơn la", "lào cai", "yên bái", "mộc châu", "mai châu",
    "ninh bình", "tam đảo", "phan thiết", "mũi né", "côn đảo", "phú quý",
    "phong nha", "bà nà", "bà nả",
]

TEXT_FEATURES = [
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


def count_destinations(texts):
    texts_concat = " ".join(texts.str.lower())
    counts = {}
    for loc in VIETNAM_DESTINATIONS:
        c = len(re.findall(re.escape(loc), texts_concat))
        if c > 0:
            counts[loc] = c
    return counts


def compute_intent_features(df_ref, df_cur):
    ref_pct = df_ref["intent"].value_counts(normalize=True).sort_index()
    cur_pct = df_cur["intent"].value_counts(normalize=True).sort_index()
    all_intents = sorted(set(ref_pct.index) | set(cur_pct.index))

    ref_arr = np.array([ref_pct.get(i, 0) for i in all_intents]) + 1e-10
    cur_arr = np.array([cur_pct.get(i, 0) for i in all_intents]) + 1e-10

    max_delta = max(abs(ref_arr - cur_arr))
    js_div = float(jensenshannon(ref_arr, cur_arr, base=2))
    # Chi-square: treat counts from a hypothetical 1000-sample distribution
    ref_n = len(df_ref)
    cur_n = len(df_cur)
    ref_counts = np.array([df_ref["intent"].value_counts().get(i, 0) for i in all_intents])
    cur_counts = np.array([df_cur["intent"].value_counts().get(i, 0) for i in all_intents])
    # Normalize to same total for chi-square
    scale = ref_n / cur_n
    cur_scaled = cur_counts * scale
    try:
        chi2_stat, chi2_p = chisquare(cur_scaled, f_exp=ref_counts)
    except Exception:
        chi2_stat, chi2_p = 0.0, 1.0

    return {
        "intent_js_divergence": round(js_div, 4),
        "intent_chi2_pvalue": round(float(chi2_p), 4),
        "intent_max_delta_pp": round(float(max_delta) * 100, 2),
        "intent_labels": all_intents,
        "intent_ref_pct": [round(ref_pct.get(i, 0) * 100, 2) for i in all_intents],
        "intent_cur_pct": [round(cur_pct.get(i, 0) * 100, 2) for i in all_intents],
    }


def compute_destination_features(df_ref, df_cur):
    dest_ref = count_destinations(df_ref["text"])
    dest_cur = count_destinations(df_cur["text"])
    all_dests = sorted(set(dest_ref.keys()) | set(dest_cur.keys()))

    ref_arr = np.array([dest_ref.get(d, 0) for d in all_dests]) + 1e-10
    cur_arr = np.array([dest_cur.get(d, 0) for d in all_dests]) + 1e-10

    total_ref = ref_arr.sum()
    total_cur = cur_arr.sum()
    ref_pct = ref_arr / total_ref if total_ref > 0 else ref_arr
    cur_pct = cur_arr / total_cur if total_cur > 0 else cur_arr

    # Entropy
    def entropy(p):
        p_norm = p / p.sum()
        return -np.sum(p_norm * np.log2(p_norm + 1e-10)) if p.sum() > 0 else 0

    ent_ref = entropy(ref_pct)
    ent_cur = entropy(cur_pct)

    # Top-1 share
    top1_ref = float(ref_arr.max() / total_ref) if total_ref > 0 else 0
    top1_cur = float(cur_arr.max() / total_cur) if total_cur > 0 else 0

    # Ha Giang spike
    ha_giang_ref = dest_ref.get("hà giang", 0)
    ha_giang_cur = dest_cur.get("hà giang", 0)

    # JS divergence of destination distribution
    dest_js = float(jensenshannon(ref_pct, cur_pct, base=2))

    return {
        "dest_entropy_ref": round(ent_ref, 4),
        "dest_entropy_cur": round(ent_cur, 4),
        "dest_top1_share_ref": round(top1_ref, 4),
        "dest_top1_share_cur": round(top1_cur, 4),
        "dest_js_divergence": round(dest_js, 4),
        "ha_giang_ref": int(ha_giang_ref),
        "ha_giang_cur": int(ha_giang_cur),
        "dest_labels": all_dests,
        "dest_ref_counts": [int(dest_ref.get(d, 0)) for d in all_dests],
        "dest_cur_counts": [int(dest_cur.get(d, 0)) for d in all_dests],
    }


def run(ref_path, cur_path):
    log.info("📊 Intent Drift Monitor — Hot Trend Scenario")
    log.info(f"  Ref: {ref_path}")
    log.info(f"  Cur: {cur_path}")

    df_ref = pd.read_csv(ref_path)
    df_cur = pd.read_csv(cur_path)

    for df in (df_ref, df_cur):
        if "raw_text" in df.columns and "text" not in df.columns:
            df.rename(columns={"raw_text": "text"}, inplace=True)
        if "intent" not in df.columns:
            log.error("  ❌ CSV must have 'intent' column")
            return

    # 1. Intent features
    log.info("\n  [1/4] Computing intent distribution features...")
    intent_feat = compute_intent_features(df_ref, df_cur)
    log.info(f"    JS divergence: {intent_feat['intent_js_divergence']:.4f}")
    log.info(f"    Chi2 p-value:  {intent_feat['intent_chi2_pvalue']:.4f}")
    log.info(f"    Max delta:     {intent_feat['intent_max_delta_pp']:.2f}pp")

    # 2. Destination features
    log.info("\n  [2/4] Computing destination distribution features...")
    dest_feat = compute_destination_features(df_ref, df_cur)
    log.info(f"    Entropy: {dest_feat['dest_entropy_ref']:.2f} → {dest_feat['dest_entropy_cur']:.2f}")
    log.info(f"    Top-1:   {dest_feat['dest_top1_share_ref']:.1%} → {dest_feat['dest_top1_share_cur']:.1%}")
    log.info(f"    Hà Giang: {dest_feat['ha_giang_ref']} → {dest_feat['ha_giang_cur']}")

    # 3. Text quality features
    log.info("\n  [3/4] Computing text quality features...")
    feat_ref = compute_features(df_ref)
    feat_cur = compute_features(df_cur)

    # 4. Evidently DataDrift on text features
    log.info("\n  [4/4] Running Evidently DataDriftPreset...")
    data_def = DataDefinition(numerical_columns=TEXT_FEATURES)
    dataset_ref = Dataset.from_pandas(feat_ref[TEXT_FEATURES], data_definition=data_def)
    dataset_cur = Dataset.from_pandas(feat_cur[TEXT_FEATURES], data_definition=data_def)

    report = Report(metrics=[DataDriftPreset(num_threshold=0.05)])
    result = report.run(reference_data=dataset_ref, current_data=dataset_cur)

    reports_dir = os.path.join(ROOT, "results", "evidently_reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(reports_dir, f"intent_drift_{ts}.html")
    result.save_html(html_path)
    log.info(f"    ✅ Evidently report: {html_path}")

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

    # Feature-level results
    feature_results = []
    for col in TEXT_FEATURES:
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

    # Composite scoring: intent (0.4) + destination (0.4) + text quality (0.2)
    intent_penalty = min(intent_feat["intent_js_divergence"] * 2, 1.0)
    dest_penalty = min(dest_feat["dest_js_divergence"] * 2, 1.0)
    text_penalties = sorted([f["penalty"] for f in feature_results], reverse=True)
    text_penalty = sum(text_penalties[:3]) / 3 if text_penalties else 0

    composite_score = 1.0 - (intent_penalty * 0.4 + dest_penalty * 0.4 + text_penalty * 0.2)
    composite_score = max(0.0, min(1.0, composite_score))

    # Log results
    log.info(f"\n{'='*60}")
    log.info(f"  INTENT DRIFT RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Intent penalty:     {intent_penalty:.2f} (weight 0.4)")
    log.info(f"  Destination penalty: {dest_penalty:.2f} (weight 0.4)")
    log.info(f"  Text quality penalty: {text_penalty:.2f} (weight 0.2)")
    log.info(f"  Composite score:     {composite_score:.4f}")
    log.info(f"  Evidently drift:    {evidently_drift_share:.2%}")

    log.info(f"\n  Intent delta:")
    for i, intent in enumerate(intent_feat["intent_labels"]):
        ref_p = intent_feat["intent_ref_pct"][i]
        cur_p = intent_feat["intent_cur_pct"][i]
        d = cur_p - ref_p
        icon = "🔴" if abs(d) > 2.0 else "🟢"
        log.info(f"    {icon} {intent:25s} {ref_p:5.1f}% → {cur_p:5.1f}%  delta={d:+.1f}pp")

    # Build distributions for text features
    distributions = {}
    for col in TEXT_FEATURES:
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

    record = {
        "scenario": "hot_trend",
        "ref_source": os.path.basename(ref_path),
        "cur_source": os.path.basename(cur_path),
        "ref_rows": len(df_ref),
        "cur_rows": len(df_cur),
        "composite_score": round(composite_score, 4),
        "intent_penalty": round(intent_penalty, 4),
        "dest_penalty": round(dest_penalty, 4),
        "text_penalty": round(text_penalty, 4),
        "evidently_drift_share": round(evidently_drift_share, 4),
        "threshold": QUALITY_THRESHOLD,
        "features": feature_results,
        "distributions": distributions,
        **intent_feat,
        **dest_feat,
        "report_html_filename": f"intent_drift_{ts}.html",
        "email_sent": False,
    }

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("intent_drift", json.dumps(record)),
    )
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"\n  ✅ Đã lưu vào mlops_reports (report_type=intent_drift)")

    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=os.path.join(ROOT, "data", "reference_normal_v3.csv"))
    parser.add_argument("--cur", default=os.path.join(ROOT, "data", "current_trend_v3.csv"))
    args = parser.parse_args()
    run(ref_path=args.ref, cur_path=args.cur)
