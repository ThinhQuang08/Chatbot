# Generate drift evaluation charts for reports.
import json
import os
import sys
import csv
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RASA_DIR = os.path.join(ROOT_DIR, "rasa_bot")
REPORT_DIR = os.path.join(ROOT_DIR, "report_output")
os.makedirs(REPORT_DIR, exist_ok=True)

INTENT_MAP = {
    "search_destination": "travel_planning",
    "search_travel": "travel_planning",
}
HELD_OUT_F1 = {
    "affirm": 0.772, "ask_itinerary": 0.857, "ask_location_feature": 0.609,
    "ask_policy_booking": 0.769, "ask_tour_info": 0.786, "ask_transportation": 0.545,
    "ask_weather_timing": 0.600, "book_tour": 0.615, "bot_challenge": 0.842,
    "deny": 0.571, "goodbye": 0.800, "greet": 0.875, "inform": 0.684,
    "out_of_scope": 0.533, "search_accommodation": 0.848, "search_activity": 0.588,
    "search_food_dining": 0.667, "search_price": 0.564, "thanks": 0.880,
    "travel_planning": 0.490,
}


def load_errors_and_report():
    rp = os.path.join(RASA_DIR, "results", "intent_report.json")
    ep = os.path.join(RASA_DIR, "results", "intent_errors.json")
    if not os.path.exists(rp):
        print("Run evaluate_drift.py first to generate results.")
        sys.exit(1)
    with open(rp) as f:
        report = json.load(f)
    errors = []
    if os.path.exists(ep):
        with open(ep) as f:
            errors = json.load(f)
    return report, errors


def plot_metrics_comparison(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    intents = sorted(k for k in report
                     if k not in ("accuracy", "macro avg", "weighted avg", "micro avg"))
    drift_f1s = [report[i]["f1-score"] for i in intents]
    held_f1s = [HELD_OUT_F1.get(i, 0) for i in intents]
    supports = [report[i]["support"] for i in intents]
    labels = [i.replace("_", " ").title() for i in intents]

    x = np.arange(len(intents))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - w / 2, drift_f1s, w, label="Drift (current_trend_v3)", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + w / 2, held_f1s, w, label="Held-out test", color="#3498db", alpha=0.85)

    for bar, f1 in zip(bars1, drift_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1-score", fontsize=13)
    ax.set_ylim(0, 1.05)
    drift_macro = report.get("macro avg", {}).get("f1-score", 0)
    drift_macro = report.get("macro avg", {}).get("f1-score", 0)
    ax.axhline(y=0.6948, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=drift_macro, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(intents) - 1, 0.72, "test = 0.695", fontsize=9, color="blue", ha="right")
    label_y = drift_macro + 0.03 if drift_macro < 0.95 else 0.96
    ax.text(len(intents) - 1, label_y, f"drift = {drift_macro:.3f}", fontsize=9, color="red", ha="right")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(REPORT_DIR, "drift_vs_test_f1.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drift_confusion_matrix(errors, intents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    label_to_idx = {l: i for i, l in enumerate(intents)}
    n = len(intents)
    cm_counts = Counter((e["intent"], e["intent_prediction"]["name"]) for e in errors)
    cm = np.zeros((n, n), dtype=int)
    for (true, pred), cnt in cm_counts.items():
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] = cnt

    # Diagonals = correct predictions
    report_path = os.path.join(RASA_DIR, "results", "intent_report.json")
    with open(report_path) as f:
        report = json.load(f)
    for intent in intents:
        i = label_to_idx[intent]
        total = report.get(intent, {}).get("support", 0)
        off_diag = cm[i, :].sum() - cm[i, i]
        cm[i, i] = total - off_diag

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.45)))
    max_val = cm.max() if cm.max() > 0 else 1
    im = ax.imshow(cm, cmap="RdYlBu_r", vmin=0, vmax=max_val)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Examples", fontsize=11)

    labels = [i.replace("_", " ").title() for i in intents]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > max_val * 0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=5, color=color)

    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "drift_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drift_error_pie(errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conf = Counter((e["intent"], e["intent_prediction"]["name"]) for e in errors)
    top = conf.most_common(8)
    labels = [f"{t} → {p}" for (t, p), c in top]
    sizes = [c for (t, p), c in top]
    other = len(errors) - sum(sizes)
    if other > 0:
        labels.append(f"Other ({other})")
        sizes.append(other)

    colors = plt.cm.tab10(range(len(labels)))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.7,
        textprops={"fontsize": 9},
    )
    plt.legend(wedges, labels, title="True → Predicted",
               loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.axis("equal")
    fig = plt.gcf()
    fig.set_size_inches(11, 6)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "drift_error_pie.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drift_metrics_card(report, n_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    acc = report.get("accuracy", 0)
    mf1 = macro.get("f1-score", 0)
    drop = 0.6948 - mf1

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    metrics = [
        ("Drift dataset", "current_trend_v3", ""),
        ("Samples", str(n_test), ""),
        ("Macro F1", f"{mf1:.4f}", f"↓ {drop:.1f} vs test"),
        ("Accuracy", f"{acc:.4f}", ""),
        ("Held-out test F1", "0.6948", "baseline"),
        ("Drift gap", f"{drop:.3f} ({drop/0.6948*100:.1f}%)", "performance drop"),
    ]

    rows = len(metrics)
    col_widths = [0.25, 0.25, 0.5]
    for row_idx, (label, value, note) in enumerate(metrics):
        y = 1 - (row_idx + 0.5) / rows
        c = "#2c3e50" if row_idx == 0 else "black"
        ax.text(0.05, y, label, fontsize=11, fontweight="bold" if row_idx == 0 else "normal",
                va="center", color=c)
        ax.text(0.35, y, value, fontsize=11, fontweight="bold" if row_idx in (2, 5) else "normal",
                va="center", color="#e74c3c" if row_idx == 5 else "#2c3e50")
        ax.text(0.60, y, note, fontsize=9, va="center", color="#7f8c8d")

    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "drift_metrics_card.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    report, errors = load_errors_and_report()
    intents = sorted(k for k in report
                     if k not in ("accuracy", "macro avg", "weighted avg", "micro avg"))

    n_test = sum(report[i]["support"] for i in intents)

    plot_metrics_comparison(report)
    plot_drift_confusion_matrix(errors, intents)
    plot_drift_error_pie(errors)
    plot_drift_metrics_card(report, n_test)

    print(f"\nAll charts saved to {REPORT_DIR}/")
    print(f"  - drift_vs_test_f1.png")
    print(f"  - drift_confusion_matrix.png")
    print(f"  - drift_error_pie.png")
    print(f"  - drift_metrics_card.png")


if __name__ == "__main__":
    main()
