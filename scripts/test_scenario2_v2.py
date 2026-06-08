# Test Kịch bản 2: Hot Trend / Destination Buzz
# Phát hiện khi 1 điểm đến chiếm 60% lưu lượng, nhiều duplicate do copy-paste trend
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
from database.db_connection import get_connection

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


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
    print_header("BUOC 1: DATA QUALITY — DataSummaryPreset tren current_trend_v2.csv")
    from scripts.data_quality import run as dq_run
    dq_run(input_path=os.path.join(DATA_DIR, "current_trend_v2.csv"),
           scenario="trend_test_quality")
    print("  ✅ Da insert data_quality vao DB")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT metrics FROM mlops_reports WHERE report_type='data_quality' "
        "AND metrics->>'scenario'='trend_test_quality' ORDER BY id DESC LIMIT 1"
    )
    m = cur.fetchone()[0]
    cur.close()
    conn.close()

    print()
    print_subheader("PHAT HIEN DUPLICATE & MISSING")
    for metric in m["metrics"]:
        name = metric["metric_name"]
        val = metric["value"]
        if isinstance(val, dict):
            if "count" in val:
                print(f"  {name}: count={val['count']}, share={val['share']:.1%}")
            elif isinstance(val, dict) and val:
                first_val = next(iter(val.values()))
                if isinstance(first_val, dict) and "value" in first_val:
                    top = sorted(val.items(), key=lambda x: x[1]["value"], reverse=True)[:5]
                    print(f"  {name}: {len(val)} values, top: {[(k, v['value']) for k, v in top]}")
                elif "null" in val:
                    continue
                else:
                    items = [(k, v) for k, v in val.items() if isinstance(v, (int, float))]
                    top = sorted(items, key=lambda x: x[1], reverse=True)[:5]
                    if top:
                        print(f"  {name}: {len(val)} values, top: {top}")
        elif isinstance(val, (int, float)):
            print(f"  {name}: {val}")
    print()


def run_text_drift():
    print_header("BUOC 2: TEXT DRIFT — PhoBERT + PCA + KS test (normal vs trend)")
    from scripts.text_drift import run as td_run
    td_run(
        ref_name="reference_normal_v2.csv",
        cur_name="current_trend_v2.csv",
        scenario="trend_test_text",
        description="Hot trend: 60% Ha Giang vs normal distribution",
        quiet=True
    )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT metrics FROM mlops_reports WHERE report_type='text_drift' "
        "AND metrics->>'scenario'='trend_test_text' ORDER BY id DESC LIMIT 1"
    )
    m = cur.fetchone()[0]
    cur.close()
    conn.close()

    print()
    print_subheader("KET QUA PHOBERT + PCA + KS TREN TUNG CHIEU")
    total_var = m.get("pca_total_variance_ratio", sum(m["pca_explained_variance_ratio"])) * 100
    n_top = min(5, len(m["components"]))
    print(f"  Tong phuong sai giai thich: {total_var:.1f}% ({m['pca_components']} components, hien thi {n_top} PC dau)")
    print(f"  Ref: {m['n_ref']} texts, Cur: {m['n_cur']} texts")
    print()
    for c in m["components"][:n_top]:
        flag = "🚨 DRIFT" if c["drift_detected"] else "🟢 ON DINH"
        print(f"  PC{c['component']+1} (var={c['explained_variance_ratio']*100:.1f}%):")
        print(f"      KS={c['ks_statistic']:.4f}, p={c['ks_p_value']:.6f} -> {flag}")
    print()
    overall = "🚨 PHAT HIEN DRIFT" if m["overall_drift_detected"] else "🟢 KHONG CO DRIFT"
    print(f"  KET LUAN: {overall} (min p={m['overall_min_p_value']})")
    print()


def show_summary():
    print_header("TONG KET KICH BAN 2: HOT TREND / DESTINATION BUZZ")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (metrics->>'scenario')
               report_type, metrics->>'scenario', metrics
        FROM mlops_reports
        WHERE metrics->>'scenario' IN ('trend_test_quality', 'trend_test_text')
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
            intent_count = next((len(x["value"].get("counts", {})) for x in m["metrics"] if "UniqueValueCount" in x["metric_name"] and "column=intent" in x["metric_name"]), "?")
            unique_texts = int(rows - dup)
            detail = f"rows={int(rows)}, dup={int(dup)}, unique_texts={unique_texts}, unique_intents={intent_count}"
            drift_str = "🔴" if int(dup) > 0 else "🟢"
        elif rtype == "text_drift":
            drift_str = "🔴" if m["overall_drift_detected"] else "🟢"
            p_vals = [c["ks_p_value"] for c in m["components"]]
            detail = f"p_min={min(p_vals):.4f}, top_KS={max(c['ks_statistic'] for c in m['components']):.4f}"
        else:
            drift_str = "?"
            detail = ""
        print(f"  {rtype:<30s} {drift_str:<10s} {detail}")

    print()
    print_separator()
    print("""
  MO TA TINH HUONG:
  Ban đau, 10 diem den đuoc hoi đeu đan (moi noi ~10%).
  Mot bo phim anh khach quay tai Ha Giang gay sot MXH.
  Dot nhien 60% cau hoi co chua "Ha Giang", đa so copy-paste.

  KET QUA:
  • DataQuality:  DuplicatedRowCount cao (copy-paste trend), UniqueValueCount giam
  • TextDrift:    PC1 phan biet manh "Ha Giang" vs cac diem den khac (KS cao)

  XU LY:
  1. Phat hien drift -> canh bao tren dashboard
  2. Kiem tra: trend tam thoi hay vinh vien?
  3. Tam thoi -> khong retrain, chi record trong DB, cho trend qua
  4. Vinh vien -> thu thap data đa dang hon cho Ha Giang -> retrain
    """)
    print_separator()


def run_all():
    print_separator("=", 72)
    print("  KIEM TRA KICH BAN 2: HOT TREND / DESTINATION BUZZ")
    print("  Phuong phap: Normal (reference) vs Ha Giang trend (current)")
    print_separator("=", 72)

    print("""
  Tinh huong: Mot bo phim quay tai Ha Giang gay sot MXH
  → 60% luong cau hoi la ve Ha Giang
  → Đa so giong nhau (copy-paste tu trend)
  → Intent bi meo: search_destination chiem ty le cao bat thuong
    """)

    run_data_quality()
    run_text_drift()
    show_summary()


if __name__ == "__main__":
    run_all()
