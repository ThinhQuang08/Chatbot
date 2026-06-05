# detect drift dữ liệu 2 mùa
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from database.db_connection import get_connection
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


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


def run(ref_name="reference_winter.csv", cur_name="current_summer.csv",
        scenario="season", description=""):
    ref_path = os.path.join(DATA_DIR, ref_name)
    cur_path = os.path.join(DATA_DIR, cur_name)

    if not os.path.exists(ref_path):
        print(f"Khong tim thay ref: {ref_path}")
        return
    if not os.path.exists(cur_path):
        print(f"Khong tim thay cur: {cur_path}")
        return

    df_ref = pd.read_csv(ref_path)
    df_cur = pd.read_csv(cur_path)

    common_cols = list(set(df_ref.columns) & set(df_cur.columns))
    df_ref = df_ref[common_cols]
    df_cur = df_cur[common_cols]

    print(f"[{scenario}] Ref: {len(df_ref)} dong, Cur: {len(df_cur)} dong, cols: {common_cols}")

    data_def = DataDefinition(
        categorical_columns=["intent"],
        text_columns=["text"]
    )
    dataset_ref = Dataset.from_pandas(df_ref, data_definition=data_def)
    dataset_cur = Dataset.from_pandas(df_cur, data_definition=data_def)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=dataset_ref, current_data=dataset_cur)

    try:
        d = result.as_dict()
    except AttributeError:
        d = result.dict()

    d = make_serializable(d)
    d["scenario"] = scenario
    d["description"] = description

    for m in d["metrics"]:
        name = m["metric_name"]
        if "DriftedColumnsCount" in name:
            share = m["value"]["share"]
            count = m["value"]["count"]
            print(f"  Drift share: {share:.0%} ({int(count)} columns drifted)")
        elif "ValueDrift" in name:
            col = m["config"]["column"]
            pval = m["value"]
            print(f"  - {col}: p_value = {pval:.4f}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("data_drift", json.dumps(d))
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Da insert data_drift [{scenario}] vao DB")


if __name__ == "__main__":
    run()
