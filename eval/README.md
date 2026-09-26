# eval

Read-only evaluation helpers outside `camdetect/` so changes here do not redeploy `swiftbackend`.

## Layer B (`layer_b.py`)

Scores **persistence**, `lin_h30`, and `xgb_h30` on a trailing hold-out window of
`traffic_prediction.v_training_set` at the **30-minute** horizon (`y_30`). Slices
are by direction (`SG_TO_MY`, `MY_TO_SG`) and by time of day (morning peak,
evening peak, other), using the peak flags already in the view.

**What a number means:** error on the Google Maps `duration_in_traffic` series.
This is skill against persistence on that series, not an independent wait-time
measurement and not "we beat Google."

### Run (live BigQuery, read-only)

```bash
cd eval
pip install -r requirements.txt
python layer_b.py --project swiftborder --holdout-days 3
```

`model_registry` is **not** updated unless you pass `--write-registry` (reserved;
not implemented). Paste results into [docs/evaluation.md](../docs/evaluation.md)
only after a successful run, in a separate commit.

### Offline tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
