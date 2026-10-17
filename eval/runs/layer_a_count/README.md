# Layer A count comparison (15 frames)

Image-level car and vehicle totals for camera 2701, on the 15-frame Roboflow v6 `test` split. These files have no bounding boxes; the notebook's box AP50 on the same split is in [camdetect/test-reports/](../../../camdetect/test-reports/README.md#box-metrics-test-split). `eval/layer_a.py` has not been run on it.

| File | What it is |
| --- | --- |
| `artifact_counts.json` | Caption totals and the five detectors on the [comparison page](https://claude.ai/artifact/5m7x6QfuGrD911BzgVGyVn), plus the `traffic_images.metadata` GCS path for each frame (1920 x 1080). |
| `local_rescore.json` | `local:yolo26s-v6-boxfix` on those originals: confidence 0.2 and overlap 0.6; confidence 0.30 with overlap off; confidence 0.30 with overlap 0.6. Produced by `eval/score_layer_a_local.py`. |
| `roboflow_v6_conf02.json` | Live `swiftbackend` `model=v6` at confidence 0.2, overlap off, 17 Oct 2026. Fifteen billed calls. The 03:15 frame's data.gov.sg bytes match the GCS object. |

Day is 07:00-18:59 SGT from the timestamp in the file name (7 day, 8 night). The narrative is [docs/evaluation.md](../../../docs/evaluation.md#layer-a--vision). The full per-frame report with the annotated images is [camdetect/test-reports/](../../../camdetect/test-reports/README.md).
