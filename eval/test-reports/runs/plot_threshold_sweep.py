"""Car recall/precision and count error vs confidence for the two fine-tuned detectors.

The sweep (confidence 0.10-0.50, cars only, IoU 0.5, no overlap suppression) is read from the
printed output of the s-vs-m cell in vehicle_detection_comparison.ipynb, for the valid and test
splits. Per-box predictions are not in the repo, so this is a 7-point trade-off curve, not a
full precision-recall curve; the full curves are BoxPR_curve.png in each eval_test/ folder.

Writes, next to this file:
  threshold_sweep.csv   one row per split, model and confidence
  threshold_sweep.png   precision-recall points, count error and count ratio vs confidence

    python eval/test-reports/runs/plot_threshold_sweep.py
"""

import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "notebooks", "vehicle_detection_comparison.ipynb")
NAMES = {"yolo26s": "YOLO-S (served)", "yolo26m": "YOLO-M"}
COLORS = {"yolo26s": "#2a78d6", "yolo26m": "#eb6834"}  # categorical slots 1, 2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
CHOSEN, SERVED = 0.30, 0.20


def parse_sweep():
    """[{split, model, conf, recall, precision, f1, count_mae, count_ratio}] from the cell output."""
    nb = json.load(open(NOTEBOOK))
    cell = next(c for c in nb["cells"] if "RUN_TAG" in "".join(c["source"]))
    text = "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))
    rows = []
    # Splits print in order (valid, then test); each model block starts "--- yolo26x | ..."
    for split_block in re.split(r"SPLIT: ", text)[1:]:
        split = split_block.split()[0]
        for m in re.finditer(r"--- (yolo26[sm]) \|.*?\[car only\]\n.*?conf\s*\n(.*?)\n\[", split_block, re.S):
            for line in m.group(2).strip().splitlines():
                conf, rec, prec, f1, mae, ratio = map(float, line.split())
                rows.append({"split": split, "model": m.group(1), "conf": conf, "recall": rec,
                             "precision": prec, "f1": f1, "count_mae": mae, "count_ratio": ratio})
    assert {(r["split"], r["model"]) for r in rows} == {(s, m) for s in ("valid", "test") for m in NAMES}, rows
    return rows


rows = parse_sweep()
with open(os.path.join(HERE, "threshold_sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def series(split, model, key):
    pts = sorted((r for r in rows if r["split"] == split and r["model"] == model), key=lambda r: r["conf"])
    return [r["conf"] for r in pts], [r[key] for r in pts], pts


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.32})
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
LS = {"test": "-", "valid": (0, (4, 2))}

# 1. Precision vs recall, one point per confidence threshold
ax = axes[0]
for f1 in (0.75, 0.80, 0.85):  # iso-F1 guides
    r = [x / 200 for x in range(140, 200)]
    p = [f1 * x / (2 * x - f1) if 2 * x > f1 else None for x in r]
    ax.plot(r, p, color=GRID, lw=1, zorder=0)
    ax.text(0.945, f1 * 0.945 / (2 * 0.945 - f1), f"F1 {f1:.2f}", fontsize=8, color=MUTED, ha="right", va="bottom")
for model in NAMES:
    for split in ("test", "valid"):
        _, rec, pts = series(split, model, "recall")
        prec = [p["precision"] for p in pts]
        ax.plot(rec, prec, ls=LS[split], color=COLORS[model], lw=2, marker="o", ms=4,
                label=f"{NAMES[model]}, {split}")
        for p in pts:
            if p["conf"] == CHOSEN:
                ax.plot(p["recall"], p["precision"], "o", ms=10, color=COLORS[model], mec=INK, mew=1.5, zorder=5)
    _, rec, pts = series("test", model, "recall")
    for p in pts:  # label the test points with their threshold
        ax.annotate(f"{p['conf']:.2f}", (p["recall"], p["precision"]), xytext=(6, -10 if model == "yolo26s" else 4),
                    textcoords="offset points", fontsize=7.5, color=COLORS[model])
ax.set_xlim(0.72, 0.95)
ax.set_ylim(0.55, 0.95)
ax.set_xlabel("Car recall")
ax.set_ylabel("Car precision")
ax.set_title("Precision vs recall (labels = confidence)", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

# 2. Count error vs confidence
ax = axes[1]
for model in NAMES:
    for split in ("test", "valid"):
        conf, mae, _ = series(split, model, "count_mae")
        ax.plot(conf, mae, ls=LS[split], color=COLORS[model], lw=2, marker="o", ms=4,
                label=f"{NAMES[model]}, {split}")
ax.axvline(CHOSEN, color=INK, lw=1, ls=":")
ax.axvline(SERVED, color=MUTED, lw=1, ls=":")
ax.text(CHOSEN, 17.5, " 0.30 chosen", fontsize=8.5, color=INK)
ax.text(SERVED, 17.5, " 0.20 served\n (+ overlap 0.6)", fontsize=8.5, color=MUTED, va="top")
ax.set_ylim(0, 18.5)
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Car count MAE per frame")
ax.set_title("Count error is lowest at 0.25-0.40", loc="left")

# 3. Count ratio vs confidence
ax = axes[2]
ax.axhline(1.0, color=INK, lw=1)
ax.text(0.505, 1.0, "  predicted = labelled", fontsize=8.5, color=INK, va="bottom", ha="right")
for model in NAMES:
    for split in ("test", "valid"):
        conf, ratio, _ = series(split, model, "count_ratio")
        ax.plot(conf, ratio, ls=LS[split], color=COLORS[model], lw=2, marker="o", ms=4)
ax.axvline(CHOSEN, color=INK, lw=1, ls=":")
ax.axvline(SERVED, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Predicted cars ÷ labelled cars")
ax.set_title("Over-count below ~0.35", loc="left")

fig.suptitle("Choosing the confidence threshold: cars, Roboflow dataset v6, 15 images per split", x=0.08,
             ha="left", fontsize=14, fontweight="bold", color=INK, y=1.04)
fig.text(0.08, 0.955, "Solid = test split, dashed = valid split. Ringed points = conf 0.30. IoU 0.5, no overlap "
         "suppression. Source: s-vs-m cell of vehicle_detection_comparison.ipynb.", fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(HERE, "threshold_sweep.png"), dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote threshold_sweep.csv ({len(rows)} rows) and threshold_sweep.png")
