# Layer A count comparison (15 frames)

Image-level car and vehicle totals for camera 2701, on the 15-frame Roboflow v6 `test` split. Chad's per-frame report, annotated frames, and box AP50 are in [eval/test-reports/](../../../eval/test-reports/README.md#box-metrics-test-split). The count JSON files have no bounding boxes; `annotations_with_predictions.json` does. `eval/la/layer_a.py` has not been run on the split. Reproducibility for the count rescoring is [`REPORT.md`](REPORT.md).

| File | What it is |
| --- | --- |
| `artifact_counts.json` | Caption totals and the five detectors on the [comparison page](https://claude.ai/artifact/5m7x6QfuGrD911BzgVGyVn), plus the `traffic_images.metadata` GCS path for each frame (1920×1080). |
| `artifact_summary.json` | MAE/bias/exact/±2 recomputed from `artifact_counts.json` (`build_layer_a_artifact_summary.py`). |
| `local_rescore.json` | `local:yolo26s-v6-boxfix` on those originals: live point (0.2 / 0.6), artifact point (0.30 / overlap off), sensitivity (0.30 / 0.6). Includes `run` provenance. |
| `roboflow_v6_conf02.json` | Live `swiftbackend` `model=v6` at confidence 0.2, overlap off. Fifteen billed calls. Includes `run` provenance. |
| `annotations_with_predictions.json` | Roboflow dataset v6 labels for all 90 images (train/valid/test), each with YOLO-S `best.pt` predictions (class, confidence, `bbox_xywh` in pixels) at confidence >= 0.01 after class-aware NMS (IoU 0.7, imgsz 1280), no overlap dedupe. Colab export, 17 Oct 2026; `predictions_source` records the settings. |
| `frames/` | Downloaded GCS originals (gitignored). |

Day is 07:00-18:59 SGT from the timestamp in the file name (7 day, 8 night). Narrative: [docs/evaluation.md](../../../docs/evaluation.md#layer-a--vision). **Re-run:** [REPORT.md §4](REPORT.md#4-reproducibility).
