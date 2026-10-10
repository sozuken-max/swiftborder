# camdetect

HTTP Cloud Function that counts vehicles on a Singapore traffic camera frame,
and attributes each one to a direction of travel across the Causeway.

Pipeline: resolve a frame from data.gov.sg's traffic-images API -> run the
Roboflow vehicle-detection workflow over it (or, with `model=local`, the
YOLO26s ONNX model shipped in the image) -> filter by confidence -> classify
each detection by direction -> return JSON or an annotated JPEG.

## Request parameters

Accepted in the query string or a JSON object body. A body value wins unless it
is missing, `null` or an empty string (so `confidence: 0` is honoured, and
`"date_time": ""` falls back to the query string). `GET`, `POST` and
`OPTIONS` are supported.

| Parameter | Default | Meaning |
|---|---|---|
| `camera_id` | `2701` | data.gov.sg camera id |
| `date_time` | now, Asia/Singapore | frame timestamp, `YYYY-MM-DDTHH:MM:SS` |
| `confidence` | `0.1` | minimum detection confidence |
| `format` | `image` | `image`, `directional` or `json` |
| `model` | `ROBOFLOW_WORKFLOW_ID` | `v4` or `v6` (Roboflow, allowlist `WORKFLOW_VERSIONS`), or `local` (in-container model, below); case-insensitive; anything else is 400 before any upstream call |

## In-container model (`model=local`)

`model=local` scores the frame inside the service with
[`models/yolo26s_v6_boxfix.onnx`](models/yolo26s_v6_boxfix.onnx) on
onnxruntime (CPU). There is no Roboflow call and no credit cost, and it works
with `ROBOFLOW_API_KEY` unset. Every response key, header and `format` is the
same as for `v4`/`v6`. `workflow_id` / `X-Workflow-Id` read
`local:yolo26s-v6-boxfix` (`LOCAL_MODEL_ID`), and each prediction also carries
`class_id`.

| | |
| --- | --- |
| Source | Colab run `v6_yolo26s_boxfix`, Ultralytics 8.4.175, from `yolo26s.pt` on Roboflow dataset v6 (polygon labels rewritten as boxes), `imgsz=1280`, `freeze=10`, AdamW `lr0=0.001`, cosine LR, `patience=50` |
| Classes | `bus`, `car`, `truck` |
| Checkpoint | `best.pt`, sha256 `de7d105b380dbe1b1f4aa962487920b6cbb3128704f337591a8043d3b38abf0e`, 20.4 MB (not in git; it lives in the team Drive under `swiftborder_runs/v6_yolo26s_boxfix/weights/`) |
| Export | `YOLO("best.pt").export(format="onnx", imgsz=[736, 1280], simplify=True)`; sha256 `dc43db2fe30ab8cf08b6d7056bd7d20d0f111f0a5ec577a51ed86cb593f65fe8`, 38 MB |
| Validation (from the checkpoint) | mAP50 0.833, mAP50-95 0.558, P 0.797, R 0.734 (Colab, best epoch). This is the run's own validation split of about 15 images, which also picked the checkpoint. It is not a held-out score or a harness result. |

How it matches Ultralytics' `model.predict(imgsz=1280)`:

- **Input 736×1280.** A 1920×1080 frame is letterboxed to 1280×720 plus 8 px
  grey (114) bands, which is the shape Ultralytics' own predict uses. A square
  1280×1280 export padded the frame further and found 41 vehicles where
  `.pt` predict found 59 on the same CAM 2702 frame. Other frame sizes are
  letterboxed into the same input.
- **Resize.** `_resize_linear` reproduces OpenCV `INTER_LINEAR`, including
  its fixed-point rounding (bit-identical on 1920×1080 → 1280×720). PIL's
  bilinear antialiases when shrinking, which lost distant vehicles. A plain
  float bilinear still flipped borderline boxes.
- **Decoding.** The checkpoint's head has `end2end=False`, so its validation
  used the one-to-many head with class-aware NMS (IoU 0.7, max 300 boxes). The
  ONNX graph is that head, and `_local_predictions` runs the same NMS in numpy.
  The NMS-free `end2end=True` export decodes a different head and was not used.
- **Parity check.** On two live 10 Oct frames, this code and Ultralytics
  `.pt` predict at `conf=0.1` gave identical counts and class splits: CAM 2701
  121 (92 car, 23 truck, 6 bus) and CAM 2702 59 (37/20/2).

**Cost on Cloud Run.** Measured locally (4 vCPU, Python 3.11), not on the
service: about 0.4-0.6 s per 1920×1080 frame warm and about 1 s for the first
frame. Peak RSS is about 380 MB with the onnxruntime memory arena off (about
625 MB with it on). The session loads lazily on the first `model=local` request,
so Roboflow requests do not pay for it. Local inference is serialised per
instance (`_local_inference_lock`). Give the service at least 1 GiB of memory,
and expect concurrent `local` requests on one instance to queue.

A missing or unloadable model file returns **503** `Local model is not
available`. No upstream inference call is made.

To ship a new model: export with the same arguments, replace the file (or set
`LOCAL_MODEL_PATH`), change `LOCAL_MODEL_ID` so responses say which weights
scored them, update `LOCAL_MODEL_CLASSES` if the classes changed, and update
the table above. `camdetect/models/**` is **not** in the trigger's
`includedFiles`, so a model-only commit does not redeploy. Touch `main.py`
in the same commit, or add the path to the trigger.

## Directions

Each detection's foot point (bottom-centre of its box) is compared against a
per-camera polyline separating the two carriageways:

- **`SG-MY`** — above the line: Singapore heading to Malaysia.
- **`MY-SG`** — on or below the line: Malaysia heading to Singapore.
- **`Unknown`** — foot point outside the line's x-range, or no line configured
  for this camera. Still included in `vehicle_count`, never attributed.

`sg_my + my_sg + unknown` always equals `vehicle_count`.

These labels are **not** the BigQuery congestion-view names. This service emits
`SG-MY` and `MY-SG`. `cam2701.v_congestion_index_10min` maps stored values
`to_JB` / `to_Woodlands` onto `SG_TO_MY` / `MY_TO_SG`. A join has to translate
them. Camera **2702** has no dividing line here, so every detection on that
camera is `Unknown`.

These are **occupancy** counts — vehicles currently visible in each
carriageway, i.e. queue depth. They are not flow counts: data.gov.sg refreshes
each camera only every minute or so, far too sparse to track a vehicle across
frames, so "how many crossed" is not derivable from a single frame.

`congestion` bands the direction's **count** (`CONGESTION_BANDS`): `Free Flow`
(<20), `Quarter Way` (20-39), `Half Way` (40-70), `Back to Back` (>70). The
same cuts apply to both directions.

`extent` (JSON only) is the vertical spread of a direction's box centres,
`(max y - min y) / frame height`; it is 0 with fewer than two detections. It
used to set the label and no longer does: on 14 v6 frames of CAM 2701 (6-7 Oct
2026) MY-SG spread was 0.61-0.68 at anything from 17 to 93 vehicles, because a
few cars at each end of the carriageway already span it. The old top band
(0.75) was above what the carriageway can reach, so `Back to Back` never fired.

Limits of this proxy, stated so reports do not over-read it:

- The count cuts were set by eye on those 14 frames, not fitted to crossing time.
- Every SG-MY count in that sample was 12 or less, so the SG-MY cuts are
  untested on a congested frame. SG-MY is the far carriageway (smaller boxes),
  so it may need lower cuts.
- Counts depend on the detector: the same frame gave 82 MY-SG on v6 and 31 on v4.
- It is occupancy in one frame, not crossing time.

## Responses

`format=image` (default) — unchanged: JPEG with green boxes and a
`Vehicles: N` banner.

`format=directional` — same frame with the dividing line drawn in white, 1 px
boxes coloured by direction (SG-MY red, MY-SG blue, Unknown grey) with no
per-box labels, and a top-left banner whose blocks are filled with the same
colours and carry each direction's count and congestion level (plus an
"Unattributed" block when any detection has no direction). The banner is the
colour key.

Both image formats carry these headers:

```
X-Vehicle-Count          total kept detections
X-Vehicle-Count-SG-MY    Singapore -> Malaysia
X-Vehicle-Count-MY-SG    Malaysia -> Singapore
X-Vehicle-Count-Unknown  unattributed
X-Congestion-SG-MY       congestion label
X-Congestion-MY-SG       congestion label
X-Source-Image           upstream frame URL
X-Frame-Datetime         timestamp used
X-Workflow-Id            Roboflow workflow that scored the frame
```

`format=json`:

```json
{
  "camera_id": "2701",
  "date_time": "2025-12-01T07:36:21",
  "source_image": "https://images.data.gov.sg/...",
  "min_confidence": 0.1,
  "workflow_id": "vehicle-detection-proejct-vvehicle-detection-proejct-6-yolo26s-t1-logic",
  "vehicle_count": 10,
  "predictions": [{ "x": 300, "y": 1000, "width": 70, "height": 50,
                    "class": "car", "confidence": 0.9, "direction": "MY-SG" }],
  "directions": {
    "available": true,
    "sg_my": { "count": 4, "congestion": "Quarter Way" },
    "my_sg": { "count": 4, "congestion": "Half Way" },
    "unknown": { "count": 2 }
  },
  "dividing_line": [[176, 1074], "..."]
}
```

## Errors

| Status | When |
| --- | --- |
| 400 | `date_time` is not a real `YYYY-MM-DDTHH:MM:SS` string, or `model` is not in the allowlist (both checked before any upstream call) |
| 404 | data.gov.sg has no frame for that camera and time |
| 500 | `ROBOFLOW_API_KEY` is not configured and `model` is not `local` (no upstream call is made) |
| 503 | `model=local` and the ONNX model cannot be loaded |
| 502 | data.gov.sg, the frame download, or Roboflow failed, or the frame is not an image |

Error bodies are `{"error": "..."}` with a generic message; upstream detail
(URLs, exception text) goes to the service log only. A frame that is not a
valid image is rejected **before** the billed Roboflow call.

## Compatibility

Changes are additive for well-formed requests: `format=image` still returns
green boxes and a `Vehicles: N` banner, existing JSON keys and headers keep
their meaning, and an unrecognised `format` falls back to the plain image.
Differences: `confidence=0` in a JSON body is now honoured (it used to fall
back to the default); a prediction without a numeric `confidence` is reported
with `confidence: 0.0`; `directions.*.extent` is new; malformed `date_time`
returns 400 and non-image frames return 502 instead of a 500 or a silent
empty result.

## Security (live service)

The deployed `swiftbackend` accepts unauthenticated calls from any origin
(recorded in [docs/inventory.md](../docs/inventory.md) and
[docs/findings.md](../docs/findings.md)). Every call can trigger a billed
Roboflow inference. The code does not check the `Authorization` header it
lists in CORS. Lock-down is a separate, approved change; set `ALLOWED_ORIGIN`
to the demo origin at minimum.

## Configuration

All via environment variables. `ROBOFLOW_API_KEY` is required and must never be
committed.

```
ROBOFLOW_API_KEY      Roboflow key (required for v4/v6 and the default; not for model=local)
ROBOFLOW_API_URL      default https://serverless.roboflow.com
ROBOFLOW_WORKSPACE    default chads-workspace-t3qcz
ROBOFLOW_WORKFLOW_ID  workflow to invoke
DEFAULT_CAMERA_ID     default 2701
DEFAULT_CONFIDENCE    default 0.1
ALLOWED_ORIGIN        CORS origin, default *
DIVIDING_LINES        JSON, overrides the built-in per-camera lines
LOCAL_MODEL_PATH      default models/yolo26s_v6_boxfix.onnx next to main.py
LOCAL_MODEL_ID        default local:yolo26s-v6-boxfix (reported as workflow_id)
```

A malformed `DEFAULT_CONFIDENCE` or `DIVIDING_LINES` logs a warning and falls
back to the default. The live service also sets `CACHE_BUCKET`; this code does
not read it, and no code in git writes to `swiftborder-frame-cache`.

### Adding a camera

Points are pixels of a reference frame and are rescaled to whatever resolution
the camera actually returns, so calibrate once against any frame and record the
size you used. Points must be ordered by ascending x; a line that is not is
ignored and every detection falls back to `Unknown`.

```json
{
  "2701": {
    "reference_size": [1920, 1080],
    "points": [[176,1074],[500,1015],[764,929],[1074,779],[1913,317]]
  }
}
```

## Cloud Build

[cloudbuild.yaml](cloudbuild.yaml) is the build for this service: `Test`
(`python:3.11-slim`), `Build` and `Push` ([Dockerfile](Dockerfile) on that same
`python:3.11-slim`, so the base is already on the build machine), `Deploy`
(`gcloud run services update --image`). The container runs
`functions-framework --target=detect` on `$PORT`. Trigger `76bbca35` points at
this file (switched in the Console on 2026-10-08). Repo changes to the trigger are recorded in
[CHANGELOG.md](../CHANGELOG.md).

| | |
| --- | --- |
| Trigger | `76bbca35-c1b4-4836-9f34-d7adda53ea17` |
| Name | `rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc` |
| Event | push to `^main$` on `sozuken-max/swiftborder` |
| Deploy | Cloud Run `swiftbackend`, `europe-west1`, function target `detect`, Dockerfile, path `camdetect` |

Step `Test` runs `pip install -r camdetect/requirements-dev.txt` and
`python -m pytest` in `camdetect/` before the image build. A failing test
stops the deploy.

`includedFiles` starts a build only when these change:

- `camdetect/main.py`, `camdetect/requirements.txt`
- `camdetect/Dockerfile`, `camdetect/cloudbuild.yaml`
- `camdetect/*.yaml`, `camdetect/*.yml`, `camdetect/*.json`, `camdetect/*.toml`

Test-only files (`camdetect/tests/**`, `pytest.ini`, `requirements-dev.txt`) and
`camdetect/README.md` do **not** start a deploy. Run pytest locally or rely on
the next `main.py` / `requirements.txt` push to exercise CI tests.

Re-check or re-apply from an exported trigger JSON (add `includedFiles`, then
import). Do not create a second trigger for `swiftbackend`.

```bash
gcloud builds triggers describe 76bbca35-c1b4-4836-9f34-d7adda53ea17 --project=swiftborder
gcloud builds triggers import --source=trigger.json --project=swiftborder
```

## Tests

Direction geometry, congestion, payload parsing, and the HTTP handler
(`tests/test_handler.py`: parameter precedence, every `format`, 400/404/500/502
paths, non-image frames, malformed predictions, `detect_frame` parity) run
offline with data.gov.sg and Roboflow mocked. No key or network is needed.
`tests/test_local_model.py` covers `model=local`: letterbox geometry, the
resize, NMS, box mapping, the handler with no key and no Roboflow call, the
503 path, and one real run of the checked-in ONNX file (about 1 s on CPU).

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Run that from this directory (or `scripts/run_tests.ps1 -Suite camdetect` from
the repo root). `requirements.txt` is the pinned Cloud Run dependency set;
`pytest` is only in `requirements-dev.txt`. A push of `main.py` or
`requirements.txt` to `main` redeploys `swiftbackend`.

## Reuse outside the handler

`detect_frame(image_bytes, camera_id, min_confidence, workflow_id)` runs
validation, inference and direction attribution on raw bytes and returns
`kept`, `summary`, `image_size` and `points`. Pass
`workflow_id=main.LOCAL_MODEL_ID` to score with the in-container model, which
is unbilled. The offline camera backfill in
[`eval/`](../eval/README.md) uses it.
