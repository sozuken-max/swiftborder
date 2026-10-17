"""Car recall/precision and count error vs confidence for the two fine-tuned detectors.

The sweep (confidence 0.10-0.50, cars only, IoU 0.5, no overlap suppression) is read from the
printed output of the s-vs-m cell in vehicle_detection_comparison.ipynb, for the valid and test
splits. Per-box predictions are not in the repo, so this is a 7-point trade-off curve, not a
full precision-recall curve; the full curves are BoxPR_curve.png in each eval_test/ folder.

Writes, next to this file:
  threshold_sweep.csv   one row per split, model and confidence
  threshold_sweep.png   test split: precision-recall points with car PR-AUC, count error and
                        count as a % of ground truth vs confidence

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


def parse_car_ap(split="test"):
    """Car AP50 (area under the full precision-recall curve) from the full-comparison cell's table."""
    nb = json.load(open(NOTEBOOK))
    cell = next(c for c in nb["cells"] if "FULL COMPARISON" in "".join(c["source"]))
    text = "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))
    table = text[text.index("ACCURACY (AP50"):text.index("OPERATING POINT")]
    ap, current = {}, None
    for line in table.splitlines():
        parts = line.split()
        if parts and parts[0] in ("valid", "test"):
            current, parts = parts[0], parts[1:]
        if current == split and parts and parts[0] in ("trained_s", "trained_m"):
            ap[{"trained_s": "yolo26s", "trained_m": "yolo26m"}[parts[0]]] = float(parts[2])  # AP50 car
    assert set(ap) == set(NAMES), ap
    return ap


rows = parse_sweep()
CAR_AP = parse_car_ap()
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
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.3})
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
SPLIT = "test"


def mark_thresholds(ax, top):
    ax.axvline(CHOSEN, color=INK, lw=1, ls=":")
    ax.axvline(SERVED, color=MUTED, lw=1, ls=":")
    ax.text(CHOSEN, top, " 0.30 chosen", fontsize=8.5, color=INK, va="top")
    ax.text(SERVED, top, " 0.20 served\n (+ overlap 0.6)", fontsize=8.5, color=MUTED, va="top")


# 1. Precision vs recall, one point per confidence threshold; PR-AUC from the full curve
ax = axes[0]
for f1 in (0.80, 0.85):  # iso-F1 guides
    r = [x / 400 for x in range(320, 381)]
    ax.plot(r, [f1 * x / (2 * x - f1) for x in r], color=GRID, lw=1, zorder=0)
    ax.text(0.928, f1 * 0.928 / (2 * 0.928 - f1), f"F1 {f1:.2f}", fontsize=8, color=MUTED, ha="right", va="bottom")
for model in NAMES:
    _, rec, pts = series(SPLIT, model, "recall")
    ax.plot(rec, [p["precision"] for p in pts], color=COLORS[model], lw=2, marker="o", ms=5,
            label=f"{NAMES[model]}: PR-AUC {CAR_AP[model]:.3f}")
    for p in pts:
        if p["conf"] == CHOSEN:
            ax.plot(p["recall"], p["precision"], "o", ms=11, color=COLORS[model], mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"{p['conf']:.2f}", (p["recall"], p["precision"]),
                    xytext=(7, -11) if model == "yolo26s" else (-26, 5),
                    textcoords="offset points", fontsize=8, color=COLORS[model])
ax.set_xlim(0.76, 0.93)
ax.set_ylim(0.60, 0.95)
ax.set_xlabel("Car recall")
ax.set_ylabel("Car precision")
ax.set_title("Car precision vs recall (labels = confidence)", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK,
          title="PR-AUC = car AP50, full curve", title_fontsize=8.5)

# 2. Count error vs confidence
ax = axes[1]
for model in NAMES:
    conf, mae, _ = series(SPLIT, model, "count_mae")
    ax.plot(conf, mae, color=COLORS[model], lw=2, marker="o", ms=5, label=NAMES[model])
    i = mae.index(min(mae))
    ax.annotate(f"{mae[i]:.2f}", (conf[i], mae[i]), xytext=(0, -15), textcoords="offset points",
                ha="center", fontsize=9, color=COLORS[model], fontweight="bold")
mark_thresholds(ax, 14.8)
ax.set_ylim(0, 15)
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Car count MAE (cars per frame)")
ax.set_title("Count error per frame", loc="left")
ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=INK)

# 3. Predicted count as a % of ground truth
ax = axes[2]
ax.axhline(100, color=INK, lw=1)
ax.text(0.085, 99, "100% = exact total", fontsize=8.5, color=INK, va="top", ha="left")
for model in NAMES:
    conf, ratio, _ = series(SPLIT, model, "count_ratio")
    pct = [x * 100 for x in ratio]
    ax.plot(conf, pct, color=COLORS[model], lw=2, marker="o", ms=5, label=NAMES[model])
    for c, v in zip(conf, pct):
        if c in (SERVED, CHOSEN):
            ax.annotate(f"{v:.0f}%", (c, v), xytext=(8, 2 if model == "yolo26s" else -12),
                        textcoords="offset points", fontsize=9, color=COLORS[model], fontweight="bold")
mark_thresholds(ax, 152)
ax.set_ylim(80, 153)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Predicted cars as % of labelled cars")
ax.set_title("Total count as % of ground truth (451 cars)", loc="left")

fig.suptitle("Choosing the confidence threshold: cars on the test split (15 frames, 451 labelled cars)",
             x=0.08, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.04)
fig.text(0.08, 0.955, "Points: the notebook's sweep at 7 thresholds (IoU 0.5, no overlap suppression). PR-AUC: area "
         "under the full car precision-recall curve from the notebook's comparison cell. Ringed = conf 0.30.",
         fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(HERE, "threshold_sweep.png"), dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote threshold_sweep.csv ({len(rows)} rows) and threshold_sweep.png; car PR-AUC {CAR_AP}")
