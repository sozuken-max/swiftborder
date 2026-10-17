"""Training curves for the two fine-tuned detectors, from each run's Ultralytics results.csv.

Writes, next to this file:
  training_curves.csv   both runs in one table (one row per model and epoch)
  training_curves.png   validation mAP on top, train and validation losses below

    python eval/test-reports/runs/plot_training_curves.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"YOLO-S (served)": "yolo26s", "YOLO-M": "yolo26m"}
COLORS = {"YOLO-S (served)": "#2a78d6", "YOLO-M": "#eb6834"}  # categorical slots 1, 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# results.csv column -> column in the combined CSV
COLUMNS = {
    "epoch": "epoch",
    "metrics/mAP50(B)": "val_mAP50",
    "metrics/mAP50-95(B)": "val_mAP50_95",
    "metrics/precision(B)": "val_precision",
    "metrics/recall(B)": "val_recall",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/l1_loss": "train_l1_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/l1_loss": "val_l1_loss",
    "lr/pg0": "lr",
}


def load(run):
    with open(os.path.join(HERE, run, "training", "results.csv")) as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    return [{new: (int(float(r[old])) if new == "epoch" else float(r[old])) for old, new in COLUMNS.items()}
            for r in rows]


def best_epoch(rows):
    # Ultralytics fitness: 0.1 * mAP50 + 0.9 * mAP50-95 picks best.pt
    return max(rows, key=lambda r: 0.1 * r["val_mAP50"] + 0.9 * r["val_mAP50_95"])


data = {name: load(run) for name, run in RUNS.items()}

with open(os.path.join(HERE, "training_curves.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "best_epoch"] + list(COLUMNS.values()))
    w.writeheader()
    for name, rows in data.items():
        best = best_epoch(rows)["epoch"]
        for r in rows:
            w.writerow({"model": RUNS[name], "best_epoch": int(r["epoch"] == best), **r})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11})
fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1], hspace=0.55, wspace=0.22)


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(ylabel)


# Top: validation mAP50 (solid) and mAP50-95 (dashed), best epoch marked
ax = fig.add_subplot(gs[0, :])
style(ax, "Validation mAP (valid split, 15 images)", "mAP")
best_notes = []
for name, rows in data.items():
    ep = [r["epoch"] for r in rows]
    c = COLORS[name]
    ax.plot(ep, [r["val_mAP50"] for r in rows], color=c, lw=2, label=f"{name} mAP50")
    ax.plot(ep, [r["val_mAP50_95"] for r in rows], color=c, lw=2, ls=(0, (4, 2)), label=f"{name} mAP50-95")
    b = best_epoch(rows)
    for key in ("val_mAP50", "val_mAP50_95"):
        ax.plot(b["epoch"], b[key], "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=5)
    best_notes.append(f"{name}: best.pt at epoch {b['epoch']}, mAP50 {b['val_mAP50']:.3f}, "
                      f"mAP50-95 {b['val_mAP50_95']:.3f}")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 1)
ax.legend(loc="lower right", ncol=2, frameon=False, labelcolor=INK)
ax.text(0.11, 0.05, "\n".join(best_notes), transform=ax.transAxes, fontsize=9, color=INK, va="bottom")

# Bottom rows: train and validation losses
LOSSES = [("box", "box loss"), ("cls", "class loss"), ("l1", "L1 box-regression loss")]
for row, split in ((1, "train"), (2, "val")):
    for col, (key, title) in enumerate(LOSSES):
        ax = fig.add_subplot(gs[row, col])
        style(ax, f"{'Train' if split == 'train' else 'Validation'} {title}")
        for name, rows in data.items():
            ax.plot([r["epoch"] for r in rows], [r[f"{split}_{key}_loss"] for r in rows],
                    color=COLORS[name], lw=1.5, label=name)
        if row == 1 and col == 0:
            ax.legend(frameon=False, labelcolor=INK)

fig.suptitle("YOLO26 fine-tuning on Roboflow dataset v6 (freeze=10, imgsz 1280)", x=0.06, ha="left",
             fontsize=14, fontweight="bold", color=INK)
fig.text(0.06, 0.935, "Per-epoch values from each run's results.csv. Training stopped early (patience 50): "
         "YOLO-S after 138 epochs, YOLO-M after 148. Dots mark the epoch kept as best.pt.",
         fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(HERE, "training_curves.png"), dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote training_curves.csv and training_curves.png")
