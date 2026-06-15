# scripts/evaluate_report.py
# Evaluate best model → full metrics + publication-quality charts.
import json
import os
import subprocess
import sys
import glob

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RASA_DIR = os.path.join(ROOT_DIR, "rasa_bot")
RESULTS_DIR = os.path.join(RASA_DIR, "results")
REPORT_DIR = os.path.join(ROOT_DIR, "report_output")

_ENV = os.environ.copy()
_PYTHONPATH = _ENV.get("PYTHONPATH", "")
if ROOT_DIR not in _PYTHONPATH.split(os.pathsep):
    _ENV["PYTHONPATH"] = os.pathsep.join(filter(None, [_PYTHONPATH, ROOT_DIR]))

os.makedirs(REPORT_DIR, exist_ok=True)


def find_best_model():
    models = sorted(
        glob.glob(os.path.join(RASA_DIR, "models", "*.tar.gz")),
        key=os.path.getctime,
    )
    if not models:
        print("No trained model found.")
        sys.exit(1)
    return models[-1]


def run_evaluation(model_path):
    print(f"Evaluating: {os.path.basename(model_path)}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for f in os.listdir(RESULTS_DIR):
        fp = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    result = subprocess.run(
        [
            "rasa", "test", "nlu",
            "--nlu", "data/test/nlu.yml",
            "--model", model_path,
            "--out", RESULTS_DIR,
        ],
        cwd=RASA_DIR, capture_output=True, text=True, env=_ENV,
    )
    if result.returncode != 0:
        print(f"Error:\n{result.stderr[:1000]}")
        sys.exit(1)


def load_report():
    with open(os.path.join(RESULTS_DIR, "intent_report.json")) as f:
        return json.load(f)


def load_errors():
    path = os.path.join(RESULTS_DIR, "intent_errors.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def plot_confusion_matrix(errors, intents, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    label_to_idx = {l: i for i, l in enumerate(intents)}
    n = len(intents)
    cm = np.zeros((n, n), dtype=int)

    for e in errors:
        true = e["intent"]
        pred = e["intent_prediction"]["name"]
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1

    # Diagonals = correct (fill with total correct per intent)
    report = load_report()
    for intent in intents:
        i = label_to_idx[intent]
        total = report.get(intent, {}).get("support", 0)
        off_diag = cm[i, :].sum() - cm[i, i]
        cm[i, i] = total - off_diag

    fig, ax = plt.subplots(figsize=(max(12, n * 0.6), max(10, n * 0.5)))
    max_val = cm.max() if cm.max() > 0 else 1
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max_val)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of examples", fontsize=11)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(intents, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(intents, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, str(val) if val > 0 else "",
                    ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_f1_chart(report_data, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    intents = sorted([
        k for k in report_data
        if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")
    ])
    names = [i.replace("_", " ").title() for i in intents]
    f1s = [report_data[i]["f1-score"] for i in intents]
    supports = [report_data[i]["support"] for i in intents]

    bars = plt.barh(range(len(intents)), f1s, color="steelblue")
    for idx, (bar, f1, sup) in enumerate(zip(bars, f1s, supports)):
        plt.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f} (n={sup})",
            va="center", fontsize=8,
        )

    plt.yticks(range(len(intents)), names, fontsize=9)
    plt.xlabel("F1-score", fontsize=12)
    plt.xlim(0, 1.05)
    plt.axvline(x=0.6948, color="red", linestyle="--", linewidth=1,
                label=f"Macro avg = {0.6948:.3f}")
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()
    fig = plt.gcf()
    fig.set_size_inches(10, max(6, len(intents) * 0.35))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_error_pie(errors, intents, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    conf = Counter((e["intent"], e["intent_prediction"]["name"]) for e in errors)
    top = conf.most_common(12)
    labels = [f"{t} → {p}" for (t, p), c in top]
    sizes = [c for (t, p), c in top]
    other = len(errors) - sum(sizes)
    if other > 0:
        labels.append("Other")
        sizes.append(other)

    colors = plt.cm.Set3(range(len(labels)))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
    )
    plt.legend(
        wedges, labels, title="True → Predicted",
        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7,
    )
    plt.axis("equal")
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_text_report(report_data, errors, model_name, test_count):
    from collections import Counter

    conf = Counter((e["intent"], e["intent_prediction"]["name"]) for e in errors)
    macro = report_data.get("macro avg", {})
    weighted = report_data.get("weighted avg", {})
    weighted = report_data.get("weighted avg", {})

    lines = []
    lines.append("=" * 64)
    lines.append(f"  NLU EVALUATION REPORT — {os.path.basename(model_name).replace('.tar.gz','')}")
    lines.append("=" * 64)
    lines.append(f"  Test set:  {test_count} examples")
    lines.append(f"  Accuracy:  {report_data.get('accuracy', 0):.4f}")
    lines.append(f"  Macro F1:  {macro.get('f1-score', 0):.4f}")
    lines.append(f"  Weighted:  {weighted.get('f1-score', 0):.4f}")
    lines.append(f"  Errors:    {len(errors)}/{test_count} ({100*len(errors)/test_count:.1f}%)")
    lines.append("")

    lines.append(f"{'Intent':28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  {'Support':>7s}")
    lines.append("-" * 56)
    for k in sorted(report_data):
        if k in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            continue
        v = report_data[k]
        lines.append(
            f"{k:28s} {v['precision']:6.3f} {v['recall']:6.3f} "
            f"{v['f1-score']:6.3f}  {v['support']:7d}"
        )
    lines.append("-" * 56)
    lines.append(
        f"{'macro avg':28s} {macro.get('precision', 0):6.3f} "
        f"{macro.get('recall', 0):6.3f} "
        f"{macro.get('f1-score', 0):6.3f}  "
        f"{macro.get('support', ''):>7}"
    )
    lines.append("")
    lines.append(f"Top 10 confusions (true → predicted):")
    lines.append(f"  {'True Intent':28s} {'→ Predicted':28s}  Count")
    lines.append(f"  {'-'*28}   {'-'*28}  {'-'*5}")
    for (true, pred), count in conf.most_common(10):
        lines.append(f"  {true:28s}   {pred:28s}  {count:5d}")
    lines.append("")
    lines.append(f"Worst 5 intents by F1:")
    worst = sorted(
        [(k, v["f1-score"]) for k, v in report_data.items()
         if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")],
        key=lambda x: x[1],
    )[:5]
    for intent, f1 in worst:
        lines.append(f"  {intent:30s} F1={f1:.3f}")
    lines.append("")
    lines.append(f"Best 5 intents by F1:")
    best = sorted(
        [(k, v["f1-score"]) for k, v in report_data.items()
         if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")],
        key=lambda x: -x[1],
    )[:5]
    for intent, f1 in best:
        lines.append(f"  {intent:30s} F1={f1:.3f}")
    lines.append("=" * 64)

    text = "\n".join(lines)
    print(text)
    txt_path = os.path.join(REPORT_DIR, "evaluation_report.txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {txt_path}")


def main():
    model_path = find_best_model()
    run_evaluation(model_path)

    report_data = load_report()
    errors = load_errors()

    intents = sorted([
        k for k in report_data
        if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")
    ])

    test_count = sum(
        report_data.get(i, {}).get("support", 0) for i in intents
    )

    # Text report
    print_text_report(report_data, errors, model_path, test_count)

    # Confusion matrix
    plot_confusion_matrix(errors, intents,
                          os.path.join(REPORT_DIR, "confusion_matrix.png"))

    # F1 bar chart
    plot_f1_chart(report_data,
                  os.path.join(REPORT_DIR, "f1_per_intent.png"))

    # Error pie
    plot_error_pie(errors, intents,
                   os.path.join(REPORT_DIR, "error_distribution.png"))

    print(f"\nAll outputs in: {REPORT_DIR}/")
    print(f"  - evaluation_report.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - f1_per_intent.png")
    print(f"  - error_distribution.png")


if __name__ == "__main__":
    main()
