# Local data (not in git)

Everything under `eval/data/` except this README and `.gitkeep` is gitignored.

| Path | Written by | Content |
| --- | --- | --- |
| `causeway_gdata.csv` | `timeseries_xgb.sync_canonical_travel_times` (`--refresh-bq`) | Full read-only export of `swiftborder.causeway.travel_times` (all routes) |
| `features_10min.parquet` | `features.py --build` | Joined causal 10-min feature table |
| `camera/cam2701_counts.csv` | `backfill_camera_counts.py` | Camera 2701 counts per sampled bin (append-only, resumable) |
| `layer_a/` | you | Drop a Roboflow export here (COCO `_annotations.coco.json` or YOLO `labels/` + `images/`) for `layer_a.py` |

Weather CSVs live in `Causeway/data/` (see [Causeway/README.md](../../Causeway/README.md)).

## Refresh the Maps export

```bash
cd eval
python -c "import timeseries_xgb as t; t.sync_canonical_travel_times('data/causeway_gdata.csv', refresh=True)"
```

Needs Application Default Credentials. Each report run records the export's SHA-256, size and observed time range in `run.json`, so the exact file a result came from is identifiable.
