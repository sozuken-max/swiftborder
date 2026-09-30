# Evaluation run outputs

Each harness run writes one folder `eval/runs/<run_id>/` with figures and a **schema v2** `run.json` ([`run_artifacts.py`](../run_artifacts.py)).

## What to commit for the final report

| Path | Git | Role |
| --- | --- | --- |
| **`report/`** | **Yes** | Canonical snapshot cited in the report (figures + `run.json` + `SOURCE_RUN.json`). |
| `<run_id>/` | No | Local experiment runs (regenerate freely). |
| `LATEST.json` | No | Local pointer to the most recent run (written by every harness). |

## `run.json` (schema v2)

| Key | Content |
| --- | --- |
| `schema_version` | `2` |
| `provenance` | `git_sha`, `git_branch`, `git_dirty`, `code_sha256` (SHA-256 of `eval/*.py` and `sql/**/*.sql`), Python and package versions |
| `components` | Any of `offline`, `bqml`, `joined`, `layer_a` |
| `<component>.dataset` | Source table or file, repo-relative cache path with SHA-256 and size, row counts, observed time range |
| `<component>.window` | Test window `start` / `end` (and timezone / basis); BQML also records model training times |
| `<component>.metrics` | Rows of `candidate`, `slice`, `n`, `mae_min`, `rmse_min` |
| `<component>.significance` | `ComparisonResult.to_dict()` rows: mean AE difference, block-bootstrap CI, Diebold–Mariano p-values, Holm p, decision. `dm_pvalue` and `dm_pvalue_reference` are one-sided (challenger better / reference better); `dm_pvalue_two_sided` is what Holm adjusts, and `holm_adjusted_p` drives `decision`. The "DM p" column in each run's `README.md` is the two-sided value. |
| `<component>.artifacts` | Figure paths relative to the run folder |

`promote_report_run.py` refuses a run that fails validation (missing keys, empty metrics, absent figures, absolute paths) or whose `code_sha256` differs from the current `eval/` and `sql/` sources. It also refuses a run from a dirty working tree unless `--allow-dirty` is passed. That override is recorded in `report/SOURCE_RUN.json` and in the `report/README.md` banner.

```bash
cd eval
python promote_report_run.py --check            # validate LATEST.json's run
python promote_report_run.py <run_id>           # copy to report/
python promote_report_run.py <run_id> --allow-dirty
```

## Regenerate

```bash
cd eval
pip install -r requirements-dev.txt
python generate_comparison_plots.py             # offline 60 min (CSV cache)
python generate_comparison_plots.py --bqml      # + BQML 30 min fixed window (read-only BigQuery)
python generate_comparison_plots.py --bqml --window-end "2026-09-30 23:50" --joined   # the report's components
```

The last line is how `report/` was produced (weather CSVs first; see [eval/README.md](../README.md#reproduce-the-report-run)).

**Run id format:** `YYYYMMDDTHHMMSSZ_<components>` (UTC).

Protocol diagrams (`eval-layer-a.png`, `eval-layer-b.png`) stay in [`docs/images/`](../../docs/images/).
