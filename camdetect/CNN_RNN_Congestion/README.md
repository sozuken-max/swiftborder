# CNN-RNN congestion research (camera 2701)

Deepak's Colab notebooks. They ask whether **pixels** carry congestion information beyond the clock, and how well that lines up with **Google Maps** duration on the causeway. This is **not** Chad's Layer A detector comparison ([test-reports/](../test-reports/README.md)) and **not** part of the live MVP: nothing here deploys to Cloud Run, writes to BigQuery, or feeds `forecast-api`.

## Where this sits in SwiftBorder

| Path | Role |
| --- | --- |
| **Production Layer A** | `swiftbackend` → YOLO / `model=local` → directional **counts** (interpretable, served to Hosting). |
| **Production Layer B** | `gmap-woodlands-fetcher` → `causeway.travel_times` → `forecast-api` → 30-minute **Maps duration** forecast (no camera in the serve path). |
| **Offline Layer B (scored)** | `eval/` harness: tabular XGBoost, BQML-era models, optional **camera counts** or **camera queue profile** as features ([evaluation.md §7](../../docs/evaluation.md#7-is-layer-a-output-a-meaningful-layer-b-input)). |
| **This folder** | **Research fork:** end-to-end **image → duration / band** (ResNet18 + BiGRU), trained and judged in Colab. Competes with the idea "detect vehicles, then use counts," not with the deployed services. |

Same data plane as the rest of the project:

- Frames: `gs://sg-lta-traffic-cameras/camera_id=2701/…` (same bucket as `traffic-backfill` and `cam2701.local_counts`).
- Labels: `swiftborder.causeway.travel_times` (Woodlands routes, joined by timestamp).
- Study window in v10a: **6 Sep 2026 02:00 – 4 Oct 2026** SGT (overlaps the Maps label window and the `local_counts` backfill).

```text
  LTA / GCS frames ──────────────┐
                                 ├──► Colab CNN-RNN notebooks (this folder)
  Maps travel_times (BigQuery) ──┘         │
                                           ▼
                                    checkpoints in /content only
                                    (not in git, not on GCP serve)

  Parallel production path (unchanged):

  GCS ──► swiftbackend (YOLO) ──► counts ──► UI
  Maps ──► travel_times ──► forecast-api ──► UI
```

## Notebooks

| File | Purpose |
| --- | --- |
| `Congestion_CNN_RNN_Google_vs_LTA_v8_residual.ipynb` | Residual models (baseline + CNN-RNN correction), image vs time-only control, validation MAE vs Google minutes. |
| `Congestion_CNN_RNN_Google_vs_LTA_v10a_second_rater.ipynb` | v8/v9 pipeline plus **blind human ratings** (1–5 congestion), **second-rater kit**, and **image-swap** test (does the model use the picture or only the clock?). |

**Experiment A (classification):** softmax CNN-RNN → nine Google duration bands (original brief; see references to `v6_clean` in the v10a intro).

**Experiment B (image information):** regression / residual heads per direction (SG→JB and JB→SG). Variants include `resid_image`, `resid_image_time`, and `resid_time_only` (no image). Picture-quality tests use `best_image`, not the overall validation winner if that winner is the no-image control.

## Why Google MAE alone is weak evidence

The notebook states the mismatch explicitly: Maps `duration_in_traffic` reflects roughly **7 km** of road including approaches the camera does not see; camera **2701** sits between the booths. A model can read the frame well and still score poorly against Google. That is why v10a adds human ratings and the swap test instead of treating Maps MAE as the only ground truth.

## Relation to other evaluation work

| Work | Question |
| --- | --- |
| [Layer A count eval](../../docs/evaluation.md#layer-a--vision) | Are **detector counts** right on labelled frames? |
| [Section 7c](../../docs/evaluation.md#7c-observed-counts-scored-2026-10-10) | Do **tabular counts** correlate with Maps duration and help XGBoost slightly? |
| [deep_forecast](../../docs/deep-learning-assessment.md) | Do **sequence models on Maps bins** beat XGBoost at 60 min? |
| **CNN-RNN notebooks** | Does the **image itself** predict congestion / Maps bands, and do humans agree? |

None of the CNN-RNN metrics are in `eval/runs/report/run.json`. Promoting them would need a harness (or a documented Colab export) and honest separation from the served stack.

## How to run

1. Open `Congestion_CNN_RNN_Google_vs_LTA_v10a_second_rater.ipynb` in **Google Colab** with a **T4 GPU**.
2. Sign in to GCP project `swiftborder` when prompted (read access to the bucket and BigQuery).
3. Follow the notebook: download frames, train or reuse checkpoints (`REUSE_CHECKPOINTS`), then run sections 9–10 if doing human evaluation.
4. Export any label CSVs and kits before the runtime ends (`/content` is ephemeral).

Weights and checkpoints are **not** checked into git.

## Practice Module techniques

Supports **supervised learning** (Google bands, human ratings), **deep learning** (CNN + RNN), and **intelligent sensing** (pixels). It is complementary to served YOLO counts and to tabular Layer B models; it does not replace them in the current architecture.

Design context for the wider system: [architecture.md](../../docs/architecture.md#offline-image-to-duration-research-cnn-rnn).
