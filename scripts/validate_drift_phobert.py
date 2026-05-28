import argparse
import json
import sys
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_phobert():
    from transformers import AutoTokenizer, AutoModel
    import torch
    print("[PHOBERT] Loading vinai/phobert-base-v2...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    model.eval()
    return tokenizer, model


def get_embeddings(texts, tokenizer, model, batch_size=64):
    import torch
    all_embs = []
    n = len(texts)
    print(f"[PHOBERT] Extracting embeddings for {n} texts...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls)
            if (i + batch_size) % 256 == 0 or i + batch_size >= n:
                print(f"   ... {min(i + batch_size, n)}/{n}")
    return np.vstack(all_embs)


def compute_centroid_distance(ref_embs, drift_embs):
    if len(ref_embs) == 0 or len(drift_embs) == 0:
        return None
    ref_centroid = np.mean(ref_embs, axis=0, keepdims=True)
    drift_centroid = np.mean(drift_embs, axis=0, keepdims=True)
    dist = float(cosine_distances(ref_centroid, drift_centroid)[0, 0])
    return round(dist, 4)


def compute_lr_confidence(ref_embs, drift_embs, ref_labels, drift_labels):
    le = LabelEncoder()
    all_labels = list(ref_labels) + list(drift_labels)
    le.fit(all_labels)

    y_ref = le.transform(ref_labels)
    y_drift = le.transform(drift_labels)

    try:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(ref_embs, y_ref)
        probs = clf.predict_proba(drift_embs)
        confidences = np.max(probs, axis=1)
        mean_conf = float(np.mean(confidences))
        predicted = le.inverse_transform(np.argmax(probs, axis=1))
        accuracy = float(np.mean(predicted == drift_labels))
        return {
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(accuracy, 4),
        }
    except Exception as e:
        print(f"[LR] Warning: {e}")
        return {"mean_confidence": None, "accuracy": None}


def compute_by_intent(ref_df, drift_df, ref_embs, drift_embs):
    report = {}
    all_intents = set(ref_df["predicted_intent"].unique()) | set(drift_df["predicted_intent"].unique())
    for intent in sorted(all_intents):
        ref_mask = ref_df["predicted_intent"] == intent
        drift_mask = drift_df["predicted_intent"] == intent
        ref_n = int(ref_mask.sum())
        drift_n = int(drift_mask.sum())
        if ref_n == 0 or drift_n == 0:
            report[intent] = {"ref_count": ref_n, "drift_count": drift_n, "drift_score": None}
            continue
        ref_emb = ref_embs[ref_mask.values]
        drift_emb = drift_embs[drift_mask.values]
        d = float(cosine_distances(
            np.mean(ref_emb, axis=0, keepdims=True),
            np.mean(drift_emb, axis=0, keepdims=True)
        )[0, 0])
        report[intent] = {
            "ref_count": ref_n,
            "drift_count": drift_n,
            "drift_score": round(d, 4),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate drift using PhoBERT embeddings")
    parser.add_argument("--input", default="data/drift_full_data.csv",
                        help="CSV with 'raw_text' and 'segment_label'(reference/drift) columns")
    parser.add_argument("--output", default="data/drift_validation_result.json",
                        help="Output JSON path")
    parser.add_argument("--text-column", default="raw_text")
    parser.add_argument("--label-column", default="predicted_intent")
    parser.add_argument("--segment-column", default="segment_label")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("[ERROR] Input CSV is empty")
        return

    print(f"[VALIDATE] Loaded {len(df)} rows from {args.input}")

    ref_df = df[df[args.segment_column] == "reference"].reset_index(drop=True)
    drift_df = df[df[args.segment_column] == "drift"].reset_index(drop=True)

    if ref_df.empty or drift_df.empty:
        print("[ERROR] Missing reference or drift segment in data")
        return

    print(f"[VALIDATE] Reference: {len(ref_df)} rows, Drift: {len(drift_df)} rows")

    tokenizer, model = load_phobert()

    all_texts = pd.concat([ref_df[args.text_column], drift_df[args.text_column]]).tolist()
    all_embs = get_embeddings(all_texts, tokenizer, model)

    ref_embs = all_embs[:len(ref_df)]
    drift_embs = all_embs[len(ref_df):]

    # ── Method 1: Centroid distance ──
    centroid_dist = compute_centroid_distance(ref_embs, drift_embs)
    print(f"[CENTROID] Cosine distance: {centroid_dist}")

    # ── Method 2: Logistic Regression confidence drop ──
    lr_result = compute_lr_confidence(
        ref_embs, drift_embs,
        ref_df[args.label_column].tolist(),
        drift_df[args.label_column].tolist(),
    )
    print(f"[LR] Mean confidence on drift: {lr_result['mean_confidence']}")
    print(f"[LR] Accuracy on drift: {lr_result['accuracy']}")

    # ── Method 3: By-intent breakdown ──
    by_intent = compute_by_intent(ref_df, drift_df, ref_embs, drift_embs)

    # ── Method 4: KL divergence of intent distribution ──
    ref_intent_dist = ref_df[args.label_column].value_counts(normalize=True)
    drift_intent_dist = drift_df[args.label_column].value_counts(normalize=True)
    all_intents = set(ref_intent_dist.index) | set(drift_intent_dist.index)
    kl_div = 0.0
    for intent in all_intents:
        p = ref_intent_dist.get(intent, 1e-10)
        q = drift_intent_dist.get(intent, 1e-10)
        kl_div += p * np.log(p / q)
    kl_div = round(float(kl_div), 4)
    print(f"[KL-DIVERGENCE] Intent distribution shift: {kl_div}")

    # ── Overall drift score (0-1) ──
    scores = []
    if centroid_dist is not None:
        scores.append(min(centroid_dist * 2, 1.0))
    if lr_result.get("mean_confidence") is not None:
        scores.append(1.0 - lr_result["mean_confidence"])
    scores.append(min(kl_div, 1.0))
    overall = round(float(np.mean(scores)), 4) if scores else 0.0

    severity = "low"
    if overall > 0.5:
        severity = "high"
    elif overall > 0.25:
        severity = "medium"

    result = {
        "drift_detected": overall > 0.2,
        "overall_drift_score": overall,
        "severity": severity,
        "methods": {
            "centroid_cosine_distance": centroid_dist,
            "lr_confidence_on_drift": lr_result["mean_confidence"],
            "lr_accuracy_on_drift": lr_result["accuracy"],
            "kl_divergence_intent": kl_div,
        },
        "by_intent": by_intent,
        "data_profile": {
            "total_rows": len(df),
            "reference_rows": len(ref_df),
            "drift_rows": len(drift_df),
            "reference_intents": int(ref_df[args.label_column].nunique()),
            "drift_intents": int(drift_df[args.label_column].nunique()),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Drift validation complete")
    print(f"   Overall drift score: {overall}")
    print(f"   Severity: {severity}")
    print(f"   Drift detected: {result['drift_detected']}")
    print(f"   Output: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
