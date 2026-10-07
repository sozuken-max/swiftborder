# camdetect

HTTP Cloud Function that counts vehicles on a Singapore traffic camera frame,
and attributes each one to a direction of travel across the Causeway.

Pipeline: resolve a frame from data.gov.sg's traffic-images API -> run the
Roboflow vehicle-detection workflow over it -> filter by confidence -> classify
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
| `model` | `ROBOFLOW_WORKFLOW_ID` | `v4` or `v6` (allowlist `WORKFLOW_VERSIONS`); anything else is 400 before any upstream call |

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

`extent` (JSON only) is the **vertical spread** of a direction's box centres,
`(max y - min y) / frame height`; it is 0 with fewer than two detections.
`congestion` bands that spread: `Free Flow` (<0.25), `Quarter Way` (<0.5),
`Half Way` (<0.75), `Back to Back` (otherwise). A direction with more than
70 detections (`CONGESTION_COUNT_MAX`) is `Back to Back` whatever its spread:
a dense queue bunched in the far half of the frame otherwise read `Half Way`.

Limits of this proxy, stated so reports do not over-read it:

- It measures spread, not how far the queue reaches: one distant vehicle gives 0.
- It uses box centres, while direction uses foot points.
- The SG-MY carriageway lies above a diagonal line, so its maximum possible
  spread is well under 1.0; bands are not comparable between directions.
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
| 500 | `ROBOFLOW_API_KEY` is not configured (no upstream call is made) |
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
ROBOFLOW_API_KEY      Roboflow key (required)
ROBOFLOW_API_URL      default https://serverless.roboflow.com
ROBOFLOW_WORKSPACE    default chads-workspace-t3qcz
ROBOFLOW_WORKFLOW_ID  workflow to invoke
DEFAULT_CAMERA_ID     default 2701
DEFAULT_CONFIDENCE    default 0.1
ALLOWED_ORIGIN        CORS origin, default *
DIVIDING_LINES        JSON, overrides the built-in per-camera lines
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

There is no `cloudbuild.yaml` in this repo. Deploy uses an inline Cloud Build
trigger in project `swiftborder` (config is not stored in git). Repo changes to
that trigger are recorded in [CHANGELOG.md](../CHANGELOG.md).

| | |
| --- | --- |
| Trigger | `76bbca35-c1b4-4836-9f34-d7adda53ea17` |
| Name | `rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc` |
| Event | push to `^main$` on `sozuken-max/swiftborder` |
| Deploy | Cloud Run `swiftbackend`, `europe-west1`, function target `detect`, buildpacks, path `camdetect` |

Step `Test` runs `pip install -r camdetect/requirements-dev.txt` and
`python -m pytest` in `camdetect/` before the buildpack step. A failing test
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

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Run that from this directory (or `scripts/run_tests.ps1 -Suite camdetect` from
the repo root). `requirements.txt` is the pinned Cloud Run dependency set;
`pytest` is only in `requirements-dev.txt`. A push of `main.py` or
`requirements.txt` to `main` redeploys `swiftbackend`.

## Reuse outside the handler

`detect_frame(image_bytes, camera_id, min_confidence)` runs validation,
inference and direction attribution on raw bytes and returns `kept`,
`summary`, `image_size` and `points`. The offline camera backfill in
[`eval/`](../eval/README.md) uses it.
