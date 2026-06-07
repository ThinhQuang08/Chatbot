# detect text drift using PhoBERT embedding + PCA + KS test
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from database.db_connection import get_connection

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_phobert(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PHOBERT] Loading vinai/phobert-base-v2 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def embed_texts(texts, tokenizer, model, device, batch_size=64):
    import torch
    all_embs = []
    n = len(texts)
    print(f"[EMBED] Encoding {n} texts (batch_size={batch_size})...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls)
            if (i + batch_size) % 256 == 0 or i + batch_size >= n:
                print(f"   ... {min(i + batch_size, n)}/{n}")
    return np.vstack(all_embs)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "value"):
        return obj.value
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def run(ref_name="reference_winter.csv", cur_name="current_summer.csv",
        scenario="text_drift", description="", device=None):
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

    if "text" not in df_ref.columns or "text" not in df_cur.columns:
        print("CSV files must have 'text' column")
        return

    texts_ref = df_ref["text"].dropna().tolist()
    texts_cur = df_cur["text"].dropna().tolist()
    texts_ref = [t for t in texts_ref if isinstance(t, str) and len(t.strip()) >= 3]
    texts_cur = [t for t in texts_cur if isinstance(t, str) and len(t.strip()) >= 3]

    print(f"[{scenario}] Ref: {len(texts_ref)} texts, Cur: {len(texts_cur)} texts")

    if len(texts_ref) < 3 or len(texts_cur) < 3:
        print("Can not compute drift (< 3 texts each)")
        return

    tokenizer, model, dev = load_phobert(device)

    all_texts = texts_ref + texts_cur
    all_embs = embed_texts(all_texts, tokenizer, model, dev)
    n_ref = len(texts_ref)

    n_components = min(5, len(all_texts) - 1)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(all_embs)

    results = {
        "scenario": scenario,
        "method": "phobert_pca_ks",
        "description": description,
        "n_ref": len(texts_ref),
        "n_cur": len(texts_cur),
        "pca_components": n_components,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": []
    }

    for comp_idx in range(n_components):
        comp_scores = scores[:, comp_idx]
        scores_ref = comp_scores[:n_ref]
        scores_cur = comp_scores[n_ref:]
        stat, p_value = ks_2samp(scores_ref, scores_cur)
        results["components"].append({
            "component": int(comp_idx),
            "explained_variance_ratio": float(pca.explained_variance_ratio_[comp_idx]),
            "ks_statistic": float(round(stat, 4)),
            "ks_p_value": float(round(p_value, 4)),
            "drift_detected": bool(p_value < 0.05),
        })

    overall_p = min(c["ks_p_value"] for c in results["components"])
    results["overall_drift_detected"] = overall_p < 0.05
    results["overall_min_p_value"] = overall_p

    print(f"  Overall min p_value: {overall_p:.6f}")
    print(f"  Drift detected: {results['overall_drift_detected']}")
    for c in results["components"]:
        flag = "🚨" if c["drift_detected"] else "🟢"
        print(f"  PC{c['component']+1} (var={c['explained_variance_ratio']:.2%}): "
              f"{flag} KS={c['ks_statistic']:.4f} p={c['ks_p_value']:.4f}")

    results = make_serializable(results)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mlops_reports (report_type, metrics) VALUES (%s, %s::jsonb)",
        ("text_drift", json.dumps(results))
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Da insert text_drift [{scenario}] vao DB")


if __name__ == "__main__":
    run()
