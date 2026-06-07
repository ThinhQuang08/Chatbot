# Test Kịch bản 1: Seasonal Intent Distribution Drift
# Chạy cả 3 phương pháp và hiển thị kết quả thống nhất
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from database.db_connection import get_connection

warnings.filterwarnings("ignore")


def print_separator(char="=", width=72):
    print(char * width)


def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()


def print_subheader(text):
    print()
    print(f"── {text} ──")
    print()


def run_data_quality():
    print_header("BƯỚC 1: DATA QUALITY — DataSummaryPreset trên nlu.yml")
    from scripts.data_quality import run as dq_run
    dq_run(scenario="seasonal_test_quality")
    print("  ✅ Da insert data_quality vao DB")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT metrics FROM mlops_reports WHERE report_type='data_quality' "
        "AND metrics->>'scenario'='seasonal_test_quality' ORDER BY id DESC LIMIT 1"
    )
    m = cur.fetchone()[0]
    cur.close()
    conn.close()

    for metric in m["metrics"]:
        name = metric["metric_name"]
        val = metric["value"]
        if isinstance(val, dict):
            if "count" in val:
                print(f"  📊 {name}: count={val['count']}, share={val['share']:.1%}")
            elif isinstance(val, dict) and val:
                first_val = next(iter(val.values()))
                if isinstance(first_val, dict) and "value" in first_val:
                    top = sorted(val.items(), key=lambda x: x[1]["value"], reverse=True)[:5]
                    print(f"  📊 {name}: {len(val)} values, top: {[(k, v['value']) for k, v in top]}")
                elif "null" in val:
                    continue
                else:
                    items = [(k, v) for k, v in val.items() if isinstance(v, (int, float))]
                    top = sorted(items, key=lambda x: x[1], reverse=True)[:5]
                    if top:
                        print(f"  📊 {name}: {len(val)} values, top: {top}")
        elif isinstance(val, (int, float)):
            print(f"  📊 {name}: {val}")
    print()


def run_text_drift():
    print_header("BƯỚC 2: TEXT DRIFT — PhoBERT + PCA + KS test (winter vs summer)")
    from scripts.text_drift import run as td_run
    td_run(
        ref_name="reference_winter.csv",
        cur_name="current_summer.csv",
        scenario="seasonal_test_text",
        description="Seasonal drift: text embedding shift via PhoBERT"
    )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT metrics FROM mlops_reports WHERE report_type='text_drift' "
        "AND metrics->>'scenario'='seasonal_test_text' ORDER BY id DESC LIMIT 1"
    )
    m = cur.fetchone()[0]
    cur.close()
    conn.close()

    print()
    print_subheader("KẾT QUẢ PHOBERT + PCA + KS TRÊN TỪNG CHIỀU")
    total_var = sum(m["pca_explained_variance_ratio"]) * 100
    print(f"  📊 Tong phuong sai giai thich: {total_var:.1f}% ({m['pca_components']} components)")
    print(f"  📊 Ref: {m['n_ref']} texts, Cur: {m['n_cur']} texts")
    print()
    for c in m["components"]:
        flag = "🚨 DRIFT" if c["drift_detected"] else "🟢 ON DINH"
        print(f"  PC{c['component']+1} (var={c['explained_variance_ratio']*100:.1f}%):")
        print(f"      KS={c['ks_statistic']:.4f}, p={c['ks_p_value']:.6f} → {flag}")
    print()
    overall = "🚨 PHAT HIEN DRIFT" if m["overall_drift_detected"] else "🟢 KHONG CO DRIFT"
    print(f"  KET LUAN: {overall} (min p={m['overall_min_p_value']})")
    print()


def show_summary():
    print_header("TONG KET KICH BAN 1: SEASONAL INTENT DISTRIBUTION DRIFT")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (metrics->>'scenario')
               report_type, metrics->>'scenario', metrics
        FROM mlops_reports
        WHERE metrics->>'scenario' IN ('seasonal_test_quality', 'seasonal_test_text')
        ORDER BY metrics->>'scenario', id DESC
    """)
    reports = cur.fetchall()
    cur.close()
    conn.close()

    print()
    print(f"  {'Phuong phap':<30s} {'Phat hien':<10s} {'Chi tiet'}")
    print(f"  {'─'*30} {'─'*10} {'─'*30}")
    for rtype, scenario, m in reports:
        if rtype == "data_quality":
            rows = next((x["value"] for x in m["metrics"] if "RowCount" in x["metric_name"]), "?")
            dup = next((x["value"] for x in m["metrics"] if "DuplicatedRowCount" in x["metric_name"]), "?")
            detail = f"rows={rows}, dup={dup}"
            drift_str = "🟢" if int(dup) == 0 else "⚠️"
        elif rtype == "text_drift":
            drift_str = "🚨" if m["overall_drift_detected"] else "🟢"
            p_vals = [c["ks_p_value"] for c in m["components"]]
            detail = f"p_min={min(p_vals):.4f}, top_KS={max(c['ks_statistic'] for c in m['components']):.4f}"
        else:
            drift_str = "?"
            detail = ""
        print(f"  {rtype:<30s} {drift_str:<10s} {detail}")

    print()
    print_separator()
    print("""
  KET LUAN:
  • DataQuality:  Kiem tra chat luong du lieu huan luyen (missing, duplicate)
  • TextDrift:    Phat hien thay doi tu vung/ngu nghia qua embedding (PhoBERT + KS)

  Neu DataQuality tot + TextDrift bao dong
  → Can thu thap du lieu mua moi, retrain model, cap nhat reference.
    """)
    print_separator()


def run_all():
    print_separator("=", 72)
    print("  KIEM TRA KICH BAN 1: SEASONAL INTENT DISTRIBUTION DRIFT")
    print("  Phuong phap: Winter (reference) vs Summer (current)")
    print_separator("=", 72)

    run_data_quality()
    run_text_drift()
    show_summary()


if __name__ == "__main__":
    run_all()
