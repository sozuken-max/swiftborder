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
| `<component>.significance` | `ComparisonResult.to_dict()` rows: mean AE difference, block-bootstrap CI, DM p, Holm p, decision |
| `<component>.artifacts` | Figure paths relative to the run folder |

`promote_report_run.py` refuses a run that fails validation (missing keys, empty metrics, absent figures, absolute paths). It also refuses a run from a dirty working tree unless `--allow-dirty` is passed. That override is recorded in `report/SOURCE_RUN.json` and in the `report/README.md` banner.

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
```

**Run id format:** `YYYYMMDDTHHMMSSZ_<components>` (UTC).

Protocol diagrams (`eval-layer-a.png`, `eval-layer-b.png`) stay in [`docs/images/`](../../docs/images/).
