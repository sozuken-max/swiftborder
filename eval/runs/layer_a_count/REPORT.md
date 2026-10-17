# Layer A count rescoring (reproducibility)

**Camera:** LTA 2701 (Woodlands checkpoint, SG side). **Metric:** image-level car and vehicle totals vs human-caption ground truth. Box AP50 on this same Roboflow v6 `test` split is in Chad's [eval/test-reports/](../../../eval/test-reports/README.md#box-metrics-test-split) (Colab IoU-0.5 scorer). This file does not recompute boxes. Directional count error and `eval/layer_a.py` are still not run.

The write-up for the module report is Chad's [eval/test-reports/](../../../eval/test-reports/README.md) (scoreboards, per-frame tables, annotated frames, box AP50). This file is how the count JSON was produced and how to re-run it. It does not replace that report.

**Report run (rescored):** 2026-10-17 UTC (`local_rescore.json` / `roboflow_v6_conf02.json` → `run.git_commit` and `run.generated_at_utc`). **Live backend:** `swiftbackend-00044-4l9` (europe-west1).

Methods summary: [docs/evaluation.md](../../../docs/evaluation.md#layer-a--vision). Claims register: [docs/findings.md](../../../docs/findings.md) finding 8.

---

## 1. Ground truth and images

| Item | Detail |
| --- | --- |
| Source | Fifteen frames with caption totals from the [Causeway detector comparison](https://claude.ai/artifact/5m7x6QfuGrD911BzgVGyVn) (camera 2701). |
| Machine-readable copy | [`artifact_counts.json`](artifact_counts.json) — same car totals as the artifact's `test_counts.csv` on all 15 images. |
| Time span | `2025-07-01T03:15:41` through `2025-09-01T19:06:18` (timestamps are SGT-style strings in filenames; hour used for day/night only). |
| Pixels scored | **1920×1080** originals in `gs://sg-lta-traffic-cameras/camera_id=2701/...` (paths in `artifact_counts.json`). The artifact UI used 1280×720 previews; rescoring always uses GCS originals. |
| Day / night | **Day** = hour 07–18 inclusive (7 frames). **Night** = other hours (8 frames). |

**Day/night split is computed in-repo; the artifact page does not publish it.**

---

## 2. Models and operating points

### 2.1 Artifact comparison (five detectors, fixed confidences)

Frozen numbers from the artifact page (not re-run in this rescoring pass). Summary recomputed from `artifact_counts.json` → [`artifact_summary.json`](artifact_summary.json).

| Model | Conf | Cars MAE | Cars bias | Vehicle MAE | Exact cars | Within ±2 cars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO-S trained (freeze) | 0.30 | 1.73 | +1.33 | 1.47 | 6/15 | 10/15 |
| YOLO-M trained (freeze) | 0.25 | 1.87 | +1.60 | 2.13 | 6/15 | 10/15 |
| Roboflow v6 | 0.30 | 3.53 | −1.40 | 4.87 | 2/15 | 10/15 |
| COCO YOLO-S @1280 | 0.10 | 11.60 | −9.87 | 9.60 | 2/15 | 5/15 |
| COCO YOLO-S @640 | 0.10 | 17.40 | −13.13 | 19.47 | 1/15 | 5/15 |

**Weights:** "YOLO-S trained (freeze)" is the same checkpoint as production **`model=local`**: `local:yolo26s-v6-boxfix` (Colab `v6_yolo26s_boxfix`, dataset v6, `freeze=10`, ONNX `camdetect/models/yolo26s_v6_boxfix.onnx`, SHA-256 `dc43db2fe30ab8cf08b6d7056bd7d20d0f111f0a5ec577a51ed86cb593f65fe8` at report run). YOLO-M is artifact-only (not in git, not served).

### 2.2 Production-aligned rescoring (same 15 GCS files)

| Setting | Role | Cars MAE | Cars bias | Day cars MAE (n=7) | Night cars MAE (n=8) | Exact cars | Within ±2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local:yolo26s-v6-boxfix`, conf **0.2**, overlap **0.6** | Served `model=local`; `cam2701.local_counts` backfill | 1.87 | +1.33 | 2.86 | 1.00 | 5/15 | 11/15 |
| Same ONNX, conf **0.30**, overlap **off** | Artifact freeze-S point (overlap off) | 1.80 | +1.27 | 2.14 | 1.50 | 6/15 | 10/15 |
| Same ONNX, conf **0.30**, overlap **0.6** | Sensitivity (not served) | 1.13 | −0.33 | 1.00 | 1.25 | 7/15 | 13/15 |
| Roboflow v6 via **live `swiftbackend`**, conf **0.2**, overlap off | Default Roboflow path when `model=v6` | 4.40 | −2.53 | 6.57 | 2.50 | 2/15 | 9/15 |

Error = predicted − ground truth. **Bias** = mean signed error (positive = over-count).

**Artifact vs ONNX at 0.30 / overlap off:** car counts match on **14/15** images. The exception is **`2025-07-22T19:57:03`**: artifact 12 cars, ONNX 11, ground truth 15.

**Colab mAP50 0.833** is the training run's `valid` split, which also chose the checkpoint. These 15 frames are the separate Roboflow v6 `test` split. Count MAE here is image-level totals. Box AP50 on the same frames is in the test report, not in this JSON.

---

## 3. Error patterns (discussion bullets for the report)

1. **Fine-tuned YOLO-S over-counts slightly** (+1.3 cars mean bias) but stays within ±2 on most frames at the served point (11/15).
2. **Roboflow v6 under-counts in dense daytime queues** (day car MAE 6.57 vs night 2.50 at conf 0.2). Worst case: **19 Jul 09:35** — GT 87 cars, live v6 58, local live 91.
3. **Overlap suppression matters** for the same ONNX at conf 0.30: overlap 0.6 lowers car MAE to 1.13 but is **not** what `local_counts` or Hosting use.
4. **COCO pretrained weights are not viable** on this set (MAE 11.6–17.4 cars).
5. **Directional counts and congestion bands** are not evaluated on these captions (totals only).

---

## 4. Reproducibility

Prerequisites: Application Default Credentials with read access to `sg-lta-traffic-cameras`; ONNX at `camdetect/models/yolo26s_v6_boxfix.onnx`; for v6 rescoring, network access to Cloud Run `swiftbackend` (15 billed Roboflow workflow calls).

```bash
# From repo root
cd eval

# 1) Frames (1920×1080) — either:
python download_layer_a_frames.py runs/layer_a_count/frames
# or gsutil -m cp $(jq -r '.images[].gcs_path' runs/layer_a_count/artifact_counts.json) runs/layer_a_count/frames/

# 2) Local ONNX (no Roboflow bill)
uv run --python 3.11 --with onnxruntime==1.29.0 --with numpy==2.4.6 --with pillow==12.3.0 \
  python score_layer_a_local.py runs/layer_a_count/frames

# 3) Live v6 (15× swiftbackend)
python score_layer_a_swiftbackend.py

# 4) Recompute artifact table from JSON
python build_layer_a_artifact_summary.py
```

Outputs:

| File | Producer |
| --- | --- |
| [`local_rescore.json`](local_rescore.json) | `score_layer_a_local.py` |
| [`roboflow_v6_conf02.json`](roboflow_v6_conf02.json) | `score_layer_a_swiftbackend.py` |
| [`artifact_summary.json`](artifact_summary.json) | `build_layer_a_artifact_summary.py` |

Each rescoring JSON includes a `run` block (UTC time, git commit, model paths, ONNX hash, `swiftbackend` revision when `gcloud` is available).

Frames are gitignored under `frames/`; only JSON results are checked in.

---

## 5. What remains out of scope for this appendix

| Item | Status |
| --- | --- |
| Box AP50 / car P/R | In Chad's test report (Colab IoU-0.5 scorer on this `test` split). Not from `eval/layer_a.py`. No mAP50-95 |
| Directional count error (2701 dividing line) | Not scored; no direction labels on these captions |
| `traffic_images.labels` in BigQuery | 0 rows; labels in Roboflow |
| v4 workflow | Not in the 15-frame set |
| Mar–Apr `cam2701` / `cam2702` bbox tables | Historical detections, unknown model version, not GT |

---

## 6. Practice module techniques

This evidence supports **supervised learning** (caption regression targets), **deep learning** (YOLO detectors), and **intelligent sensing** (LTA frames → occupancy counts). It does **not** by itself demonstrate that counts predict crossing duration; that is Layer B ([evaluation.md §7c](../../../docs/evaluation.md#7c-observed-counts-scored-2026-10-10)).
