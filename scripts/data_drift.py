# detect drift dữ liệu 2 mùa
import sys, os, json, warnings, requests
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

    is_drift_detected = False

    for m in d["metrics"]:
        name = m["metric_name"]
        if "DriftedColumnsCount" in name:
            share = m["value"]["share"]
            count = m["value"]["count"]
            print(f"  Drift share: {share:.0%} ({int(count)} columns drifted)")
            if share >= 0.5:
                is_drift_detected = True
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

    if is_drift_detected:
        print(">> Phat hien Data Drift vuot nguong 50%!")
        trigger_jenkins()

def trigger_jenkins():
    jenkins_url = os.getenv("JENKINS_URL") # e.g. http://10.0.1.5:8080
    job_name = os.getenv("JENKINS_JOB_NAME")
    user = os.getenv("JENKINS_USER")
    api_token = os.getenv("JENKINS_API_TOKEN")
    build_token = os.getenv("JENKINS_BUILD_TOKEN")

    if not all([jenkins_url, job_name, user, api_token, build_token]):
        print("Thieu cau hinh Jenkins trong .env, bo qua trigger.")
        return

    # API Trigger: POST http://JENKINS_URL/job/JOB_NAME/build?token=BUILD_TOKEN
    trigger_url = f"{jenkins_url.rstrip('/')}/job/{job_name}/build?token={build_token}"
    
    try:
        print(f"Triggering Jenkins Pipeline: {job_name} ...")
        response = requests.post(trigger_url, auth=(user, api_token), timeout=10)
        if response.status_code in [200, 201]:
            print("✅ Da kich hoat Jenkins Pipeline thanh cong!")
        else:
            print(f"❌ Kich hoat that bai: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Loi ket noi den Jenkins: {e}")

if __name__ == "__main__":
    run()
