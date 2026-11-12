"""Render the historical validation metrics without claiming a checkpoint evaluation."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    with (ROOT / "train_yolo/results.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    series = [
        ("metrics/precision(B)", "Precision", "#0d9488"),
        ("metrics/recall(B)", "Recall", "#6366f1"),
        ("metrics/mAP50(B)", "mAP50", "#2563eb"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#e09b24"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    for key, label, color in series:
        ax.plot(epochs, [float(row[key]) for row in rows], label=label, color=color, linewidth=2)
    ax.set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1), xlim=(1, max(epochs)))
    ax.set_title("Historical YOLO training log", loc="left", fontweight="bold", fontsize=18, pad=20)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(0.09, 0.025, "Source: train_yolo/results.csv | 83 epochs | Checkpoint association unverified", color="#64748b", fontsize=10)
    fig.tight_layout(rect=(0.025, 0.06, 0.995, 1))
    output = ROOT / "docs/assets/training_metrics.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
