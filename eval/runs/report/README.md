> **Committed report snapshot.** Ephemeral runs stay gitignored; this folder is the citation target for the final report. Promoted from `20260926T073406Z_offline-bqml`.

# Eval run `20260926T073406Z_offline-bqml`

Created (UTC): 2026-09-26T07:34:27Z

## Offline (sklearn XGB, 60 min)

- **Table:** `swiftborder.causeway.travel_times`
- **Cache:** `D:/Git Repositories/swiftborder/eval/data/causeway_gdata.csv` (refreshed: False)
- **Rows:** 11842 canonical; 5921 after route `jb_to_woodlands`
- **Models:** sklearn.XGBRegressor (eval/timeseries_xgb.train_xgb)
- **Horizon:** 60 min

### Figures (`offline/`)

- `offline/backtest-mae.png`
- `offline/holdout-sample.png`
- `offline/holdout-mae-diff.png`

## BQML serve audit (30 min)

- **Project:** `swiftborder`
- **View:** `traffic_prediction.v_training_set`
- **Hold-out:** 3 days
- **Window:** 2026-09-23 07:00:00+00:00 .. 2026-09-26 07:00:00+00:00
- **Models:** Persistence (y_persistence), lin_h30, xgb_h30

### Figures (`bqml/`)

- `bqml/mae-by-direction.png`
- `bqml/mae-diff-ci.png`

Machine-readable metadata: [`run.json`](run.json).
