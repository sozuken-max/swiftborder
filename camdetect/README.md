# camdetect

HTTP Cloud Function that counts vehicles on a Singapore traffic camera frame,
and attributes each one to a direction of travel across the Causeway.

Pipeline: resolve a frame from data.gov.sg's traffic-images API -> run the
Roboflow vehicle-detection workflow over it -> filter by confidence -> classify
each detection by direction -> return JSON or an annotated JPEG.

## Request parameters

Accepted in the query string or a JSON body (body wins). `GET`, `POST` and
`OPTIONS` are supported.

| Parameter | Default | Meaning |
|---|---|---|
| `camera_id` | `2701` | data.gov.sg camera id |
| `date_time` | now, Asia/Singapore | frame timestamp, `YYYY-MM-DDTHH:MM:SS` |
| `confidence` | `0.1` | minimum detection confidence |
| `format` | `image` | `image`, `directional` or `json` |

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

`congestion` reports how far up the frame a direction's detections reach, as a
fraction of frame height: `Free Flow` (<0.25), `Quarter Way` (<0.5),
`Half Way` (<0.75), `Back to Back` (otherwise).

## Responses

`format=image` (default) — unchanged: JPEG with green boxes and a
`Vehicles: N` banner.

`format=directional` — same frame with the dividing line drawn in white, boxes
coloured by direction (SG-MY red, MY-SG blue, Unknown grey), and a banner
carrying both counts and congestion levels.

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
```

`format=json`:

```json
{
  "camera_id": "2701",
  "date_time": "2025-12-01T07:36:21",
  "source_image": "https://images.data.gov.sg/...",
  "min_confidence": 0.1,
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

## Compatibility

Every change is additive. The default `format=image` response is byte-identical
to the previous version, all pre-existing JSON keys and headers keep their old
values, existing prediction fields are untouched, and an unrecognised `format`
still falls back to the legacy image. `Access-Control-Expose-Headers` is now
set, so browsers can finally read the `X-*` headers cross-origin — previously
they were sent but invisible to `fetch()`.

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

There is no `cloudbuild.yaml` in this repo. The deploy is an inline trigger in
project `swiftborder`, updated with `gcloud` on 26 Sep 2026 (not by a file in
git).

| | |
| --- | --- |
| Trigger | `76bbca35-c1b4-4836-9f34-d7adda53ea17` |
| Name | `rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc` |
| Event | push to `^main$` on `sozuken-max/swiftborder` |
| Deploy | Cloud Run `swiftbackend`, `europe-west1`, function target `detect`, buildpacks, path `camdetect` |

`includedFiles` limits the trigger to code and config under `camdetect/`:

- `camdetect/*.py`, `camdetect/**/*.py`
- `camdetect/requirements.txt`, `camdetect/requirements-dev.txt`
- `camdetect/pytest.ini`, `camdetect/Dockerfile`, `camdetect/cloudbuild.yaml`
- `camdetect/*.yaml`, `camdetect/**/*.yaml`, `camdetect/*.yml`, `camdetect/**/*.yml`
- `camdetect/*.json`, `camdetect/**/*.json`, `camdetect/*.toml`

A commit that only changes `camdetect/README.md`, or only files outside this
folder, does not start the build. The build still has no test step.

Re-check or re-apply from an exported trigger JSON (add `includedFiles`, then
import). Do not create a second trigger for `swiftbackend`.

```bash
gcloud builds triggers describe 76bbca35-c1b4-4836-9f34-d7adda53ea17 --project=swiftborder
gcloud builds triggers import --source=trigger.json --project=swiftborder
```

## Tests

Direction, congestion, and payload parsing are covered without calling
data.gov.sg or Roboflow.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Run that from this directory. `requirements.txt` stays the Cloud Run set.
`pytest` is only in `requirements-dev.txt`.

Not covered here: the HTTP handler, image download, and the Roboflow call.
Those need network and `ROBOFLOW_API_KEY`.
