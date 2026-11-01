"""Build README figures from local dataset counts and saved experiment artifacts.

Requires matplotlib and the local datasets/ and model-comparison/ directories.
Copies saved model outputs unchanged; does not run training or inference.
"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
MODELS = ("yolo11n", "yolo11s", "yolo11m", "yolo11l")
COLORS = ("#2878b5", "#008573", "#b5671d", "#8055ad")
INK = "#16324f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": INK,
                     "text.color": INK, "savefig.facecolor": "white"})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def card(ax, x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                              linewidth=0, facecolor=color))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(ROOT / "model-comparison" / "comparison.csv")
    summary = []
    for split in ("train", "valid", "test"):
        folder = ROOT / "datasets" / split
        images = [p for p in (folder / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        counts = [0, 0]
        labels = list((folder / "labels").glob("*.txt"))
        empty = 0
        for label in labels:
            lines = label.read_text(encoding="utf-8").splitlines()
            empty += not any(line.strip() for line in lines)
            for line in lines:
                if line.strip():
                    counts[int(line.split()[0])] += 1
        summary.append({"split": split, "images": len(images),
                        "normal_person": counts[0],
                        "potentially_dangerous_person": counts[1],
                        "empty_labels": empty})
    (OUT / "dataset-counts.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "model-comparison" / "comparison.csv", OUT / "model-comparison.csv")

    fig, ax = plt.subplots(figsize=(14, 3.5))
    fig.patch.set_facecolor("#0c1d30")
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(0.025, 0.86, "COMPUTER VISION / RESEARCH PROTOTYPE", color="#62d7c3", fontsize=12)
    ax.text(0.025, 0.62, "Dangerous Person Detection", color="white", fontsize=30, weight="bold")
    ax.text(0.025, 0.44, "Person-level YOLO11 detection for image and video review", color="#cfdeed", fontsize=15)
    for x, title, value in [(0.04, "DATASET", f"{sum(r['images'] for r in summary):,} images"),
                            (0.37, "EXPERIMENTS", "4 YOLO11 variants"),
                            (0.70, "DEMO", "Hugging Face Spaces")]:
        card(ax, x, 0.045, 0.27, 0.22, "#18334d")
        ax.text(x + 0.01, 0.19, title, color="#62d7c3", fontsize=10)
        ax.text(x + 0.01, 0.085, value, color="white", fontsize=14, weight="bold")
    fig.savefig(OUT / "project-banner.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.text(0.01, 0.91, "From dataset to human review", fontsize=21, weight="bold")
    stages = [("01 / DATA", f"{sum(r['images'] for r in summary):,} images", "Preserved source splits\nTwo person classes"),
              ("02 / TRAIN", "YOLO11 n / s / m / l", "640 px model input\nCompare validation metrics"),
              ("03 / INFER", "Images + video", "Person bounding boxes\nClass labels and scores"),
              ("04 / REVIEW", "Human interpretation", "Inspect model outputs\nCheck errors and omissions")]
    for i, (title, subtitle, detail) in enumerate(stages):
        x = 0.02 + i * 0.25
        card(ax, x, 0.14, 0.205, 0.57, "#edf4f8")
        ax.text(x + 0.01, 0.59, title, color=COLORS[i], fontsize=12, weight="bold")
        ax.text(x + 0.01, 0.43, subtitle, fontsize=11, weight="bold")
        ax.text(x + 0.01, 0.23, detail, fontsize=10, linespacing=1.7)
        if i < 3:
            ax.annotate("", xy=(x + 0.235, 0.42), xytext=(x + 0.213, 0.42),
                        arrowprops={"arrowstyle": "->", "color": INK, "lw": 2})
    save(fig, "system-workflow.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    splits = ["Train", "Validation", "Test"]
    for ax, keys, title in [(axes[0], ["images"], "Images per split"),
                            (axes[1], ["normal_person", "potentially_dangerous_person"], "Labeled people per split")]:
        bottom = [0, 0, 0]
        for j, key in enumerate(keys):
            values = [r[key] for r in summary]
            bars = ax.bar(splits, values, bottom=bottom, color=COLORS[j], width=0.55,
                          label=key.replace("_", " "))
            ax.bar_label(bars, labels=[f"{n:,}" for n in values],
                         label_type="edge" if j else "center",
                         color=INK if j else "white", padding=3 if j else 0, fontsize=10)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_title(title, loc="left", pad=18)
        ax.set_ylim(0, max(bottom) * 1.22)
        ax.set_ylabel("Count")
        if len(keys) > 1:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
    save(fig, "dataset-overview.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, key in zip(axes.flat, ["Precision", "Recall", "mAP50", "mAP50-95"]):
        bars = ax.bar([r["Model"] for r in rows], [float(r[key]) for r in rows], color=COLORS, width=0.55)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set(title=key, ylim=(0, 1.08), ylabel="Validation score")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    fig.suptitle("Reported validation comparison", fontsize=20, weight="bold")
    save(fig, "model-comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, key, title, unit in [(axes[0], "Inference (ms/ảnh)", "Reported inference time", "ms / image"),
                                  (axes[1], "Weights (MB)", "Checkpoint size", "MiB (CSV column: MB)")]:
        bars = ax.barh([r["Model"] for r in rows], [float(r[key]) for r in rows], color=COLORS)
        ax.bar_label(bars, fmt="%.2f", padding=5)
        ax.set(title=title, xlabel=unit)
        ax.invert_yaxis(); ax.margins(x=0.25)
    save(fig, "model-cost.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for model, color in zip(MODELS, COLORS):
        history = read_csv(ROOT / "model-comparison" / f"{model}-fit-within-640" / "results.csv")
        for ax, key in zip(axes, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]):
            ax.plot([int(r["epoch"]) for r in history], [float(r[key]) for r in history],
                    label=f"YOLO11{model[-1]} ({len(history)} epochs)", color=color, lw=1.6)
    for ax, title in zip(axes, ["Validation mAP50", "Validation mAP50-95"]):
        ax.set(title=title, xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        ax.grid(alpha=0.15); ax.legend(fontsize=8, loc="lower right", frameon=False)
    save(fig, "training-history.png")

    source = ROOT / "model-comparison" / "yolo11s-fit-within-640"
    files = {"results.png": "yolo11s-training-results.png",
             "BoxPR_curve.png": "yolo11s-pr-curve.png",
             "BoxF1_curve.png": "yolo11s-f1-curve.png",
             "confusion_matrix_normalized.png": "yolo11s-confusion-matrix.png",
             "val_batch0_labels.jpg": "yolo11s-val-batch0-labels.jpg",
             **{f"val_batch{i}_pred.jpg": f"yolo11s-val-batch{i}-pred.jpg" for i in range(3)}}
    manifest = []
    for original, copied in files.items():
        shutil.copyfile(source / original, OUT / copied)
        manifest.append({"file": copied, "source": (source / original).relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256((OUT / copied).read_bytes()).hexdigest()})
    (OUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated 6 figures and copied {len(files)} original outputs into {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
