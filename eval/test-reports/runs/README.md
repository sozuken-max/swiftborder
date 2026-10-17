# Layer A training and comparison runs

Training runs, Ultralytics evaluations and the five-detector comparison behind the
[test report](../README.md). Copied on 17 Oct 2026 from the team Drive folder
[`swiftborder_runs`](https://drive.google.com/drive/folders/1G_jBEMYGorHnfOEvcHse4pXJ2yRdv7HG)
and regrouped by model. All of it was produced by
[`vehicle_detection_comparison.ipynb`](../notebooks/vehicle_detection_comparison.ipynb)
on Roboflow dataset v6 (train 60 / valid 15 / test 15 images).

Nothing here is used by the service. The served model is the ONNX export in
[`services/camdetect/models/`](../../../services/camdetect/models/); nothing under `eval/` is in the Docker image.

```
runs/
├── yolo26s/                     YOLO26s fine-tuned (freeze=10): the served model, model=local
│   ├── training/                was v6_yolo26s_boxfix/
│   │   ├── weights/best.pt      checkpoint behind services/camdetect/models/yolo26s_v6_boxfix.onnx
│   │   ├── args.yaml            training settings
│   │   ├── results.csv/.png     loss and metrics per epoch
│   │   ├── labels.jpg           label statistics of the training set
│   │   ├── train_batch*.jpg     augmented training batches
│   │   └── Box*_curve.png, confusion_matrix*.png, val_batch*.jpg
│   │                            end-of-training check of best.pt on the valid split
│   └── eval_test/               was val_yolo26s_test_boxfix/: Ultralytics val on the test split
├── yolo26m/                     YOLO26m fine-tuned the same way; compared, not served
│   ├── training/                was v6_yolo26m_boxfix/ (same layout)
│   └── eval_test/               was val_yolo26m_test_boxfix/
└── comparison_5_detectors/      every detector on the 15 test frames
    ├── full_comparison_summary.csv   AP50 per class, recall/precision sweep, count error (valid and test)
    ├── test_counts.csv               car and vehicle counts per detector and frame
    ├── ground_truth/            labelled boxes drawn on each test frame
    ├── coco_s_640/              stock COCO YOLO26s at 640 px
    ├── coco_s_1280/             stock COCO YOLO26s at 1280 px
    ├── roboflow_v6/             Roboflow-hosted workflow v6
    │   └── api_responses/       raw API replies, <split>_2701_<timestamp>.json (valid and test)
    ├── yolo26s/                 the fine-tuned YOLO26s (trained_s in the notebook)
    └── yolo26m/                 the fine-tuned YOLO26m (trained_m in the notebook)
```

Each detector folder holds its 15 annotated test frames at 1920 x 1080, named
`2701_<YYYYMMDD>T<HHMMSS>.jpg` like [`../frames/`](../frames/README.md).
The CSVs and the notebook use the Roboflow export names (`..._jpg.rf.<hash>.jpg`); the frames
manifest maps one to the other.

## Training

| Run | Base | Image size | Batch | Epochs run (patience 50) | Best epoch | valid mAP50 | valid mAP50-95 | `best.pt` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `yolo26s/training` | `yolo26s.pt` | 1280 | 8 | 138 of 150 | 88 | 0.833 | 0.558 | 20,389,957 bytes, sha256 `de7d105b...abf0e` |
| `yolo26m/training` | `yolo26m.pt` | 1280 | 4 | 148 of 150 | 98 | 0.837 | 0.572 | 44,104,281 bytes |

Both: AdamW, `lr0=0.001`, cosine LR, `freeze=10`, seed 0, labels converted from polygons to boxes
first (the `boxfix` in the old names). Best epoch and its scores are read from `results.csv`. The
`valid` split also picked the checkpoint, so these two columns are not held-out scores. The YOLO-S
`best.pt` hash matches the checkpoint recorded for the served ONNX file in the
[camdetect README](../../../services/camdetect/README.md#in-container-model-modellocal). `last.pt` (the final epoch) was left
on Drive.

## Ultralytics evaluation on the test split

`eval_test/` is `model.val()` on the 15 test frames (546 labelled boxes), the frames in the test
report. Numbers from the notebook output; the folders hold only the plots.

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO-S (`yolo26s/eval_test`) | 0.813 | 0.791 | 0.836 | 0.562 |
| YOLO-M (`yolo26m/eval_test`) | 0.883 | 0.770 | 0.872 | 0.583 |

The notebook's own comparison scorer gives mAP50 0.827 (S) and 0.862 (M) on the same frames
([test report](../README.md#box-metrics-test-split)); the two scorers match boxes
differently.

## Left out

- `val_yolo26s_valid_boxfix/`, `val_yolo26m_valid_boxfix/` on Drive: `model.val()` on the valid
  split, which repeats the end-of-training check already in each `training/` folder.
- `weights/last.pt` for both runs (20 MB and 44 MB): the final epoch, not used anywhere.
