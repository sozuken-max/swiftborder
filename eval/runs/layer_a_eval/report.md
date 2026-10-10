# Layer A → Layer B: Comprehensive Evaluation

**Data:** `cam2701.local_counts` joined with `traffic_prediction.v_training_set`  
**Joined rows:** 18 folds over 2026-09-13 to 2026-09-30  
**Camera coverage threshold:** 60%  
**Folds with sufficient camera coverage:** 18/18  

---

## Q1: Correlation — Layer A counts vs Google Maps travel time

> Does the camera queue depth track Maps congestion?

### Direction: SG_TO_MY
  `cam_sg_my` → `y_persistence`: Pearson r=+0.462 (p=0.0000) Spearman ρ=+0.514 **
  `cam_sg_my` → `y_30`: Pearson r=+0.395 (p=0.0000) Spearman ρ=+0.505 **
  `cam_my_sg` → `y_persistence`: Pearson r=+0.357 (p=0.0000) Spearman ρ=+0.472 **
  `cam_my_sg` → `y_30`: Pearson r=+0.383 (p=0.0000) Spearman ρ=+0.501 **
  `cam_total` → `y_persistence`: Pearson r=+0.494 (p=0.0000) Spearman ρ=+0.577 **
  `cam_total` → `y_30`: Pearson r=+0.494 (p=0.0000) Spearman ρ=+0.587 **
  `cam_dir` → `y_persistence`: Pearson r=+0.462 (p=0.0000) Spearman ρ=+0.514 **
  `cam_dir` → `y_30`: Pearson r=+0.395 (p=0.0000) Spearman ρ=+0.505 **

### Direction: MY_TO_SG
  `cam_sg_my` → `y_persistence`: Pearson r=+0.057 (p=0.0003) Spearman ρ=+0.328 **
  `cam_sg_my` → `y_30`: Pearson r=+0.070 (p=0.0000) Spearman ρ=+0.302 **
  `cam_my_sg` → `y_persistence`: Pearson r=+0.618 (p=0.0000) Spearman ρ=+0.718 **
  `cam_my_sg` → `y_30`: Pearson r=+0.588 (p=0.0000) Spearman ρ=+0.681 **
  `cam_total` → `y_persistence`: Pearson r=+0.594 (p=0.0000) Spearman ρ=+0.711 **
  `cam_total` → `y_30`: Pearson r=+0.570 (p=0.0000) Spearman ρ=+0.673 **
  `cam_dir` → `y_persistence`: Pearson r=+0.618 (p=0.0000) Spearman ρ=+0.718 **
  `cam_dir` → `y_30`: Pearson r=+0.588 (p=0.0000) Spearman ρ=+0.681 **

### Direction: BOTH
  `cam_sg_my` → `y_persistence`: Pearson r=+0.264 (p=0.0000) Spearman ρ=+0.419 **
  `cam_sg_my` → `y_30`: Pearson r=+0.236 (p=0.0000) Spearman ρ=+0.404 **
  `cam_my_sg` → `y_persistence`: Pearson r=+0.484 (p=0.0000) Spearman ρ=+0.579 **
  `cam_my_sg` → `y_30`: Pearson r=+0.483 (p=0.0000) Spearman ρ=+0.579 **
  `cam_total` → `y_persistence`: Pearson r=+0.543 (p=0.0000) Spearman ρ=+0.631 **
  `cam_total` → `y_30`: Pearson r=+0.531 (p=0.0000) Spearman ρ=+0.621 **
  `cam_dir` → `y_persistence`: Pearson r=+0.470 (p=0.0000) Spearman ρ=+0.547 **
  `cam_dir` → `y_30`: Pearson r=+0.438 (p=0.0000) Spearman ρ=+0.530 **

---

## Q2: Ablation — Does Layer A improve Layer B (XGBoost) MAE?

Rolling-origin daily folds, same window as `eval/joined.py` (13–30 Sep 2026).
Camera features: `cam_dir`, `cam_opposite`, `cam_total`, `cam_frames`.
Maps-only features: `y_persistence`, `congestion_ratio`, `speed_kmh`, direction indicator.

### Aggregate results (across all folds)

| Model | Mean MAE (min) |
|---|---|
| Persistence | 2.6718 |
| XGBoost maps-only (all folds) | 2.8445 |
| XGBoost maps-only (cam-covered folds only) | 2.8445 |
| XGBoost maps+camera (cam-covered folds only) | 2.7618 |
| **Δ (camera − maps-only)** | **-0.0827 min ↓** |

### Significance (cam-covered folds only)

Paired fold t-test (camera MAE − maps MAE): t=-3.0885, p=0.006667, n=18 folds  
Mean delta: -0.0827 min

> [!NOTE]
> This is a fold-level paired t-test (not DM), reported for orientation only.
> Formal DM significance requires row-level paired errors.

---

## Interpretation

### Correlation findings
- `cam_sg_my` vs `y_persistence` (SG_TO_MY): Pearson r=+0.462 — moderate, statistically significant
- `cam_my_sg` vs `y_persistence` (SG_TO_MY): Pearson r=+0.357 — moderate, statistically significant
- `cam_sg_my` vs `y_persistence` (MY_TO_SG): Pearson r=+0.057 — weak, statistically significant
- `cam_my_sg` vs `y_persistence` (MY_TO_SG): Pearson r=+0.618 — strong, statistically significant

### Ablation findings

Camera features produced a **0.0827 min improvement** in mean MAE on 18 folds with ≥60% camera coverage.
The improvement is statistically significant at α=0.05 (paired-t p=0.0067).

### Important caveats

- Camera frames cover only ~1 frame/10 min. Finer coverage would sharpen the ablation.
- The label (`y_30`) is a future Maps estimate, not a ground-truth crossing time.
- A fold-level t-test on 18 folds is underpowered; row-level DM is needed for formal confirmation.
- `cam_dir` is the direction-matched count (SG_TO_MY → `cam_sg_my`). The opposite direction is included as a control feature.
- No Layer A frames were available after 4 Oct 2026; the 1–19 Oct protected window has no camera data and cannot be scored.