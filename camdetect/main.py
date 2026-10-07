import base64
import datetime
import io
import json
import logging
import math
import os
import re

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8
    from backports.zoneinfo import ZoneInfo

import functions_framework
import requests
from PIL import Image, ImageDraw, ImageFont

# --- Configuration (override any of these with Cloud Run environment variables) ---
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "chads-workspace-t3qcz")
ROBOFLOW_WORKFLOW_ID = os.environ.get(
    "ROBOFLOW_WORKFLOW_ID",
    "vehicle-detection-proejct-vvehicle-detection-proejct-6-yolo26s-t1-logic",
)
TRAFFIC_IMAGES_API = "https://api.data.gov.sg/v1/transport/traffic-images"
DEFAULT_CAMERA_ID = os.environ.get("DEFAULT_CAMERA_ID", "2701")
logger = logging.getLogger("camdetect")


def _env_float(name, default):
    """Float env var; a malformed value logs a warning instead of crashing the cold start."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default
    return value if math.isfinite(value) else default


DEFAULT_CONFIDENCE = _env_float("DEFAULT_CONFIDENCE", 0.1)
DATE_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
BOX_COLOR = (0, 255, 0)

# --- Directional counting ---
# Labels follow the direction of travel: SG-MY is Singapore-bound -> Malaysia,
# MY-SG is Malaysia -> Singapore.
DIR_SG_MY = "SG-MY"
DIR_MY_SG = "MY-SG"
DIR_UNKNOWN = "Unknown"

DIR_COLORS = {
    DIR_SG_MY: (255, 0, 0),
    DIR_MY_SG: (0, 0, 255),
    DIR_UNKNOWN: (160, 160, 160),
}

# Polyline separating the two carriageways, per camera. Points are pixels of a
# reference frame and are rescaled to whatever the camera actually returns, so a
# resolution change upstream does not silently move the line. Points must be
# ordered by ascending x. A vehicle's foot point (bottom-centre of its box)
# above the line is SG-MY; on or below it is MY-SG. A foot point outside the
# line's x-range is Unknown: still counted in vehicle_count, never attributed.
DEFAULT_DIVIDING_LINES = {
    "2701": {
        "reference_size": [1920, 1080],
        "points": [[176, 1074], [500, 1015], [764, 929], [1074, 779], [1913, 317]],
    }
}


def _load_dividing_lines():
    """Allow the whole per-camera line config to be replaced via env var."""
    raw = os.environ.get("DIVIDING_LINES")
    if not raw:
        return DEFAULT_DIVIDING_LINES
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("ignoring malformed DIVIDING_LINES; using the built-in lines")
        return DEFAULT_DIVIDING_LINES
    if not isinstance(parsed, dict):
        logger.warning("ignoring DIVIDING_LINES that is not a JSON object; using the built-in lines")
        return DEFAULT_DIVIDING_LINES
    return parsed


DIVIDING_LINES = _load_dividing_lines()

# Fraction of frame height spanned by a direction's detections, and the label
# each band maps to. Ordered by ascending threshold; the last entry is the
# fallback for anything above the final threshold.
CONGESTION_BANDS = ((0.25, "Free Flow"), (0.5, "Quarter Way"), (0.75, "Half Way"))
CONGESTION_MAX = "Back to Back"

# Named-workflow endpoint used by the Roboflow inference SDK.
WORKFLOW_URL = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"

# Workflow versions a caller may pick with ``model=`` (e.g. the page's side-by-side panel).
# A fixed allowlist: the request can never name an arbitrary workflow on the billed key.
WORKFLOW_VERSIONS = {
    "v4": "vehicle-detection-proejct-vvehicle-detection-proejct-4-yolo26s-t1-logic",
    "v6": "vehicle-detection-proejct-vvehicle-detection-proejct-6-yolo26s-t1-logic",
}


def _workflow_url(workflow_id=None):
    return f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{workflow_id or ROBOFLOW_WORKFLOW_ID}"

# Headers browsers may read cross-origin. Without this the X-* headers are
# invisible to fetch()/XHR, which is why they are exposed explicitly.
EXPOSED_HEADERS = ",".join(
    (
        "X-Vehicle-Count",
        "X-Vehicle-Count-SG-MY",
        "X-Vehicle-Count-MY-SG",
        "X-Vehicle-Count-Unknown",
        "X-Congestion-SG-MY",
        "X-Congestion-MY-SG",
        "X-Source-Image",
        "X-Frame-Datetime",
        "X-Workflow-Id",
    )
)


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Expose-Headers": EXPOSED_HEADERS,
        "Access-Control-Max-Age": "3600",
    }


def _error(message, status):
    headers = {**_cors_headers(), "Content-Type": "application/json"}
    return (json.dumps({"error": message}), status, headers)


def _get_camera_image_url(camera_id, date_time):
    """Return the image URL for a camera at a given SGT timestamp, or None."""
    resp = requests.get(TRAFFIC_IMAGES_API, params={"date_time": date_time}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for item in data.get("items", []) or []:
        for camera in item.get("cameras", []) or []:
            if str(camera.get("camera_id")) == str(camera_id):
                return camera.get("image")
    return None


class InvalidImage(ValueError):
    """The downloaded source frame is not a decodable image."""


def _download_image(image_url):
    """Fetch the source frame; raises requests errors on HTTP failure, InvalidImage if not an image."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    image_bytes = resp.content
    _image_size(image_bytes)
    return image_bytes


def _image_size(image_bytes):
    """(width, height) from the header, or InvalidImage."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            return probe.size
    except (OSError, ValueError) as exc:
        raise InvalidImage(str(exc)) from exc


def _run_workflow(image_bytes, workflow_id=None):
    """Post the frame to the Roboflow workflow and return the outputs list."""
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "use_cache": True,
        "enable_profiling": False,
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("ascii"),
            }
        },
    }
    resp = requests.post(
        _workflow_url(workflow_id),
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("outputs", [])


def _extract_predictions(outputs):
    """Match the Colab structure, but degrade to an empty list if it differs.

    Non-dict entries are dropped. A non-numeric confidence becomes 0.0 so filtering never raises.
    """
    try:
        preds = outputs[0]["predictions"]["predictions"]
    except (IndexError, KeyError, TypeError):
        return []
    if not isinstance(preds, list):
        return []
    clean = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        try:
            conf = float(p.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        clean.append({**p, "confidence": conf if math.isfinite(conf) else 0.0})
    return clean


def _box(pred):
    """(x1, y1, x2, y2) for a prediction, or None when geometry is missing or not numeric."""
    try:
        cx, cy = float(pred["x"]), float(pred["y"])
        w, h = float(pred["width"]), float(pred["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return None
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _dividing_line(camera_id, image_size):
    """Return this camera's polyline scaled to image_size, or None."""
    config = DIVIDING_LINES.get(str(camera_id))
    if not config or not image_size:
        return None
    try:
        points = [(float(x), float(y)) for x, y in config["points"]]
        ref_w, ref_h = (float(v) for v in config["reference_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(points) < 2 or ref_w <= 0 or ref_h <= 0:
        return None

    width, height = image_size
    sx, sy = width / ref_w, height / ref_h
    scaled = [(x * sx, y * sy) for x, y in points]
    # Interpolation assumes ascending x; reject anything that is not.
    if any(b[0] < a[0] for a, b in zip(scaled, scaled[1:])):
        return None
    return scaled


def _y_on_line(x, points):
    """Linear interpolation of the polyline at x, or None if x is off its span."""
    if x < points[0][0] or x > points[-1][0]:
        return None
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x <= x2:
            if x2 == x1:
                return y2
            return y1 + (x - x1) / (x2 - x1) * (y2 - y1)
    return points[-1][1]


def _classify_direction(pred, points):
    """Compare a prediction's foot point against the dividing polyline."""
    if not points:
        return DIR_UNKNOWN
    try:
        foot_x = float(pred["x"])
        foot_y = float(pred["y"]) + float(pred["height"]) / 2
    except (KeyError, TypeError, ValueError):
        return DIR_UNKNOWN
    y_limit = _y_on_line(foot_x, points)
    if y_limit is None:
        return DIR_UNKNOWN
    return DIR_SG_MY if foot_y < y_limit else DIR_MY_SG


def _congestion_extent(y_centres, image_height):
    """Vertical spread of a direction's box centres as a fraction of frame height (0 if <2 boxes)."""
    if not y_centres or not image_height:
        return 0.0
    return (max(y_centres) - min(y_centres)) / image_height


def _congestion_level(y_centres, image_height):
    """Queue depth label from the vertical spread of a direction's detections (see README)."""
    extent = _congestion_extent(y_centres, image_height)
    for threshold, label in CONGESTION_BANDS:
        if extent < threshold:
            return label
    return CONGESTION_MAX


def _summarize_directions(predictions, points, image_size):
    """Tag each prediction in place and return the per-direction summary."""
    height = image_size[1] if image_size else 0
    # Counts are kept apart from the centroids they are measured from: a
    # prediction with an unreadable y still has to land in a bucket, so that
    # sg_my + my_sg + unknown always equals vehicle_count.
    counts = {DIR_SG_MY: 0, DIR_MY_SG: 0, DIR_UNKNOWN: 0}
    centres = {DIR_SG_MY: [], DIR_MY_SG: [], DIR_UNKNOWN: []}

    for pred in predictions:
        direction = _classify_direction(pred, points)
        pred["direction"] = direction
        counts[direction] += 1
        try:
            centres[direction].append(float(pred["y"]))
        except (KeyError, TypeError, ValueError):
            pass

    return {
        "available": bool(points),
        "sg_my": {
            "count": counts[DIR_SG_MY],
            "congestion": _congestion_level(centres[DIR_SG_MY], height),
            "extent": round(_congestion_extent(centres[DIR_SG_MY], height), 4),
        },
        "my_sg": {
            "count": counts[DIR_MY_SG],
            "congestion": _congestion_level(centres[DIR_MY_SG], height),
            "extent": round(_congestion_extent(centres[DIR_MY_SG], height), 4),
        },
        "unknown": {"count": counts[DIR_UNKNOWN]},
    }


def _direction_headers(summary):
    return {
        "X-Vehicle-Count-SG-MY": str(summary["sg_my"]["count"]),
        "X-Vehicle-Count-MY-SG": str(summary["my_sg"]["count"]),
        "X-Vehicle-Count-Unknown": str(summary["unknown"]["count"]),
        "X-Congestion-SG-MY": summary["sg_my"]["congestion"],
        "X-Congestion-MY-SG": summary["my_sg"]["congestion"],
    }


def _load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_banner(draw, image, text):
    font = _load_font(max(18, image.width // 60))
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, box[2] - box[0] + 16, box[3] - box[1] + 12], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)


def _draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 500)
    font = _load_font(max(14, image.width // 90))

    for pred in predictions:
        box = _box(pred)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        label = f"{pred.get('class', 'vehicle')} {pred.get('confidence', 0):.2f}"
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=BOX_COLOR)
        draw.text((x1 + 3, ly + 2), label, fill=(0, 0, 0), font=font)

    _draw_banner(draw, image, f"Vehicles: {len(predictions)}")


def _draw_direction_banner(draw, image, segments):
    """Top-left banner of side-by-side blocks, each filled with its direction's box colour.

    ``segments`` is a list of (direction, text). The fill doubles as the colour key, so the
    boxes need no per-box labels.
    """
    font = _load_font(max(18, image.width // 60))
    x = 0
    for direction, text in segments:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 12
        draw.rectangle([x, 0, x + w, h], fill=DIR_COLORS[direction])
        draw.text((x + 8, 6), text, fill=(255, 255, 255), font=font)
        x += w


def _draw_directional(image, predictions, points, summary):
    """Same frame: boxes outlined 1 px in the direction colour, the divider drawn in, and a banner.

    Boxes carry no per-box labels (in heavy traffic they hid the vehicles). The banner's blocks
    are filled with the same colours, so it is both the count and the colour key.
    """
    draw = ImageDraw.Draw(image)

    if points:
        draw.line(points, fill=(255, 255, 255), width=max(3, image.width // 384))

    unknown = 0
    for pred in predictions:
        direction = pred.get("direction", DIR_UNKNOWN)
        if direction not in (DIR_SG_MY, DIR_MY_SG):
            unknown += 1
        color = DIR_COLORS.get(direction, DIR_COLORS[DIR_UNKNOWN])
        box = _box(pred)
        if box is None:
            continue
        draw.rectangle(box, outline=color, width=1)

    segments = [
        (DIR_SG_MY, f"SG \u2192 MY: {summary['sg_my']['count']} ({summary['sg_my']['congestion']})"),
        (DIR_MY_SG, f"MY \u2192 SG: {summary['my_sg']['count']} ({summary['my_sg']['congestion']})"),
    ]
    if unknown:
        segments.append((DIR_UNKNOWN, f"Unattributed: {unknown}"))
    _draw_direction_banner(draw, image, segments)


class MissingApiKey(RuntimeError):
    """ROBOFLOW_API_KEY is not configured."""


def detect_frame(image_bytes, camera_id=DEFAULT_CAMERA_ID, min_confidence=None, workflow_id=None):
    """Run detection and direction attribution on one frame (no HTTP request object).

    Returns a dict: ``kept`` (predictions at or above ``min_confidence``, each tagged with
    ``direction``), ``summary`` (per-direction counts, congestion labels and extents),
    ``image_size`` and ``points`` (the scaled dividing line or None).

    Raises InvalidImage for undecodable bytes (checked before any billed inference call),
    MissingApiKey when no key is configured, and requests/ValueError errors from the workflow.
    ``workflow_id`` defaults to ``ROBOFLOW_WORKFLOW_ID``.
    Used by ``detect()`` and by the offline camera backfill in ``eval/``.
    """
    if min_confidence is None:
        min_confidence = DEFAULT_CONFIDENCE
    image_size = _image_size(image_bytes)
    if not ROBOFLOW_API_KEY:
        raise MissingApiKey("ROBOFLOW_API_KEY is not set")
    predictions = _extract_predictions(_run_workflow(image_bytes, workflow_id))
    kept = [p for p in predictions if p["confidence"] >= min_confidence]
    points = _dividing_line(camera_id, image_size)
    summary = _summarize_directions(kept, points, image_size)
    return {"kept": kept, "summary": summary, "image_size": image_size, "points": points}


def _param(body, args, name):
    """Body wins over query string; a missing, null or empty-string body value falls back."""
    value = body.get(name) if isinstance(body, dict) else None
    if value is None or value == "":
        value = args.get(name)
    return value


def _parse_confidence(value):
    if value is None or value == "":
        return DEFAULT_CONFIDENCE
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return conf if math.isfinite(conf) else DEFAULT_CONFIDENCE


def _valid_date_time(value):
    """True for a string 'YYYY-MM-DDTHH:MM:SS' that is a real calendar time."""
    if not isinstance(value, str) or not DATE_TIME_PATTERN.match(value):
        return False
    try:
        datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return True


@functions_framework.http
def detect(request):
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())

    args = request.args or {}
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}

    camera_id = str(_param(body, args, "camera_id") or DEFAULT_CAMERA_ID)
    date_time = _param(body, args, "date_time")
    output_format = str(_param(body, args, "format") or "image").lower()
    min_confidence = _parse_confidence(_param(body, args, "confidence"))
    model = _param(body, args, "model")
    if model is None or model == "":
        workflow_id = ROBOFLOW_WORKFLOW_ID
    elif isinstance(model, str) and model.lower() in WORKFLOW_VERSIONS:
        workflow_id = WORKFLOW_VERSIONS[model.lower()]
    else:
        return _error(f"model must be one of: {', '.join(sorted(WORKFLOW_VERSIONS))}", 400)

    # If no timestamp is supplied, use the current Singapore-local time.
    if date_time is None or date_time == "":
        date_time = datetime.datetime.now(ZoneInfo("Asia/Singapore")).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    elif not _valid_date_time(date_time):
        return _error("date_time must be YYYY-MM-DDTHH:MM:SS (Singapore time)", 400)

    if not ROBOFLOW_API_KEY:
        logger.error("ROBOFLOW_API_KEY is not set")
        return _error("Service is not configured", 500)

    # 1. Resolve the source image URL from data.gov.sg
    try:
        image_url = _get_camera_image_url(camera_id, date_time)
    except (requests.RequestException, ValueError) as exc:
        logger.warning("traffic-images lookup failed: %s", exc)
        return _error("Traffic camera API request failed", 502)
    if not image_url:
        return _error(f"No image found for camera {camera_id} at {date_time}", 404)

    # 2. Download and validate the source frame (before any billed inference)
    try:
        image_bytes = _download_image(image_url)
    except requests.RequestException as exc:
        logger.warning("source frame download failed: %s", exc)
        return _error("Failed to download source image", 502)
    except InvalidImage as exc:
        logger.warning("source frame is not an image: %s", exc)
        return _error("Source frame is not a valid image", 502)

    # 3. Run the Roboflow workflow and attribute directions
    try:
        result = detect_frame(image_bytes, camera_id, min_confidence, workflow_id)
    except requests.RequestException as exc:
        logger.warning("inference request failed: %s", exc)
        return _error("Inference request failed", 502)
    except ValueError as exc:
        logger.warning("inference returned a non-JSON response: %s", exc)
        return _error("Inference returned an invalid response", 502)
    kept, summary, points = result["kept"], result["summary"], result["points"]

    # Optional JSON mode for inspecting the raw detections
    if output_format == "json":
        payload = {
            "camera_id": camera_id,
            "date_time": date_time,
            "source_image": image_url,
            "min_confidence": min_confidence,
            "workflow_id": workflow_id,
            "vehicle_count": len(kept),
            "predictions": kept,
            "directions": summary,
            "dividing_line": [[round(x, 2), round(y, 2)] for x, y in points]
            if points
            else None,
        }
        headers = {**_cors_headers(), "Content-Type": "application/json"}
        return (json.dumps(payload), 200, headers)

    # 5. Draw the boxes on the full-resolution frame and return it
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if output_format == "directional":
        _draw_directional(image, kept, points, summary)
    else:
        _draw_boxes(image, kept)

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    headers = {
        **_cors_headers(),
        **_direction_headers(summary),
        "Content-Type": "image/jpeg",
        "X-Vehicle-Count": str(len(kept)),
        "X-Source-Image": image_url,
        "X-Frame-Datetime": date_time,
        "X-Workflow-Id": workflow_id,
        "Cache-Control": "no-store",
    }
    return (out.getvalue(), 200, headers)
