# Handoff: camera 2701 count backfill (Roboflow pilot)

**Owner:** the team member who holds the Roboflow workspace key (`chads-workspace-t3qcz`). **Not** run by the Evaluation & Risk owner or an agent: the key is not shared, and the run spends the team's free-tier credits.
**Status:** code and tests done ([`eval/backfill_camera_counts.py`](../eval/backfill_camera_counts.py)); pilot **not run**. Tracked as Task 9 in [plan-eval-integrity.md](plan-eval-integrity.md).
**Why it matters:** the joined experiment ([evaluation.md §2](evaluation.md#2-joined-features-30-minutes-evaljoinedpy)) has a "Maps + weather + camera" row that stays `pending` until camera counts exist for 5–30 Sep. It is the only test of the proposal's claim that queue depth helps the forecast: the Layer A output already in BigQuery covers 13 Mar–22 Apr 2026 and does not overlap the Maps label ([evaluation.md §7](evaluation.md#7-is-layer-a-output-a-meaningful-layer-b-input)).

## What the script does

For each sampled 10-minute bin (SGT) it asks data.gov.sg for the camera 2701 frame, downloads it, and runs `camdetect.detect_frame` (the same Roboflow workflow `swiftbackend` calls). One sampled bin = **one billed Roboflow call**. Results append to `eval/data/camera/cam2701_counts.csv` (gitignored). A missing or stale frame is recorded as missing, never as zero vehicles, and costs no inference.

The Roboflow free tier has no overage billing: when included credits run out, calls fail until the monthly reset. The script stops at the first quota or payment error (HTTP 402/403, or 429 mentioning quota) with everything so far saved.

## Steps

Windows PowerShell, from the repo root, Python 3.11.

1. **Set up** (once):
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\pip install -r camdetect/requirements-dev.txt -r eval/requirements-dev.txt
   cd eval
   ```
2. **Dry run** (no requests, no credits):
   ```powershell
   ..\.venv\Scripts\python.exe backfill_camera_counts.py --dry-run
   ```
   Expect `peak-first: 2730 bins planned` for 5–30 Sep (`--mode full`: 3,743).
3. **Record the Roboflow Credit Usage page** (workspace settings → Credit Usage): credits used this cycle, credits remaining, reset date.
4. **Pilot: 50 calls on one day.** Set the key in this shell only; never write it to a file in the repo:
   ```powershell
   $env:ROBOFLOW_API_KEY = "<key>"
   ..\.venv\Scripts\python.exe backfill_camera_counts.py --max-calls 50 --start 2026-09-20 --end 2026-09-20
   ..\.venv\Scripts\python.exe backfill_camera_counts.py --coverage
   ```
5. **Record the Credit Usage page again** and compute credits per call = (after − before) / 50.
6. **Decide the budget.** Calls affordable this cycle = remaining credits / credits per call. Leave headroom for demos of `swiftbackend`, which is public and draws on the same credits ([findings.md](findings.md#known-risk-public-swiftbackend-documented-not-changed)). Then run with a cap:
   ```powershell
   ..\.venv\Scripts\python.exe backfill_camera_counts.py --max-calls <budget>
   ```
   Peak hours (06:00–10:59, 16:00–21:59 SGT) are scored first. Re-running resumes where it stopped, including after the monthly reset. Exit code 3 means a quota stop.
7. **Clear the key:** `Remove-Item env:ROBOFLOW_API_KEY`.

## What to report back

Post in the team channel, or append to the Task 9 row in [plan-eval-integrity.md](plan-eval-integrity.md):

| Item | Value |
| --- | --- |
| Credits before / after pilot | |
| Credits per call | |
| Calls made in total, and whether it stopped on quota | |
| `--coverage` totals (ok / recorded) | |
| Date and who ran it | |

Share the CSV `eval/data/camera/cam2701_counts.csv` with the evaluation owner (it is gitignored; send the file, do not commit it).

## After the backfill (evaluation owner)

Put the shared CSV at `eval/data/camera/cam2701_counts.csv`, commit any code changes first (promotion refuses a dirty tree), then:

```powershell
cd eval
..\.venv\Scripts\python.exe generate_comparison_plots.py --bqml --window-end "2026-09-30 23:50" --joined
..\.venv\Scripts\python.exe promote_report_run.py --check
..\.venv\Scripts\python.exe promote_report_run.py <run_id>
```

Do not add `--refresh-bq`: it re-downloads `travel_times` past the report cut, so the offline numbers would change for a reason unrelated to the camera.

`joined.py` scores the "+camera" models only on test rows with a camera value and reports coverage. Below 60% of test rows the row is marked **insufficient** and must not be quoted as a result. Then update the camera row in [evaluation.md](evaluation.md) and findings item 5 in [findings.md](findings.md).

## Do not

- Commit or paste the key anywhere in the repo (`.env` files are gitignored, but prefer the shell variable).
- Read the key from the Cloud Run service configuration.
- Run without `--max-calls` before the pilot has measured the cost.
