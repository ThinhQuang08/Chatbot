import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd
from database.db_connection import get_connection
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataSummaryPreset

warnings.filterwarnings("ignore")

NLU_PATH = os.path.join(os.path.dirname(__file__), "..", "rasa_bot/data/train/nlu.yml")


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


def parse_nlu_to_df(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    records = []
    for item in data.get("nlu", []):
        intent = item.get("intent")
        examples = item.get("examples", "")
        for line in examples.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                records.append({"text": line[2:], "intent": intent})
    return pd.DataFrame(records)


def run(input_path=None, scenario="nlu_training"):
    if input_path:
        df = pd.read_csv(input_path)
        print(f"Doc {len(df)} dong tu CSV: {input_path}")
    else:
        df = parse_nlu_to_df(NLU_PATH)
        print(f"Doc {len(df)} examples tu NLU file")

    data_def = DataDefinition(
        text_columns=["text"],
        categorical_columns=["intent"]
    )
    dataset = Dataset.from_pandas(df, data_definition=data_def)
    report = Report(metrics=[DataSummaryPreset()])
    result = report.run(reference_data=None, current_data=dataset)

    try:
        d = result.as_dict()
    except AttributeError:
        d = result.dict()

    d = make_serializable(d)
    d["scenario"] = scenario
    d["source"] = input_path or "nlu.yml"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("data_quality", json.dumps(d))
    )
    conn.commit()
    cur.close()
    conn.close()
    print("Da insert data_quality vao DB")


if __name__ == "__main__":
    run()
