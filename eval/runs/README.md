# Evaluation run outputs

Comparison plots and metadata from [`generate_comparison_plots.py`](../generate_comparison_plots.py) or `layer_b.py --plots`.

## What to commit for the final report

| Path | Git | Role |
| --- | --- | --- |
| **`report/`** | **Yes** | Canonical snapshot cited in the report (figures + `run.json`). |
| `<run_id>/` | No | Local / CI experiment runs (regenerate freely). |
| `LATEST.json` | Yes | Pointer to the most recent local run (dev convenience). |

Promote a good run into `report/` when the numbers in [`docs/evaluation.md`](../../docs/evaluation.md) match:

```bash
cd eval
python promote_report_run.py              # uses LATEST.json
python promote_report_run.py 20260926T073406Z_offline-bqml
git add runs/report
```

`report/SOURCE_RUN.json` records which ephemeral run was copied. Re-promote when you intentionally refresh the report baseline (note the date in evaluation prose).

**Why not commit every run?** Timestamped folders multiply quickly; the report needs **one** frozen evidence bundle, not every grid search. Row counts and BQ hold-out windows are in `run.json` so graders need not re-query GCP.

## Layout

```text
eval/runs/
  LATEST.json
  README.md
  report/                  # committed report sources
    run.json
    README.md
    SOURCE_RUN.json
    offline/*.png
    bqml/*.png
  <run_id>/                # gitignored
    ...
```

**Run id format:** `YYYYMMDDTHHMMSSZ_<components>` (UTC).

## Regenerate

```bash
cd eval
pip install -r requirements-dev.txt
python generate_comparison_plots.py --bqml
```

Protocol diagrams (`eval-layer-a.png`, `eval-layer-b.png`) stay in [`docs/images/`](../../docs/images/).
