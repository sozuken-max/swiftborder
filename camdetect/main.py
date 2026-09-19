import base64
import datetime
import io
import json
import os
from zoneinfo import ZoneInfo

import functions_framework
import requests
from PIL import Image, ImageDraw, ImageFont

# --- Configuration (override any of these with Cloud Run environment variables) ---
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "chads-workspace-t3qcz")
ROBOFLOW_WORKFLOW_ID = os.environ.get(
    "ROBOFLOW_WORKFLOW_ID",
    "vehicle-detection-proejct-vvehicle-detection-proejct-4-yolo26s-t1-logic",
)
TRAFFIC_IMAGES_API = "https://api.data.gov.sg/v1/transport/traffic-images"
DEFAULT_CAMERA_ID = os.environ.get("DEFAULT_CAMERA_ID", "2701")
DEFAULT_CONFIDENCE = float(os.environ.get("DEFAULT_CONFIDENCE", "0.1"))
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
        return DEFAULT_DIVIDING_LINES
    return parsed if isinstance(parsed, dict) else DEFAULT_DIVIDING_LINES


DIVIDING_LINES = _load_dividing_lines()

# Fraction of frame height spanned by a direction's detections, and the label
# each band maps to. Ordered by ascending threshold; the last entry is the
# fallback for anything above the final threshold.
CONGESTION_BANDS = ((0.25, "Free Flow"), (0.5, "Quarter Way"), (0.75, "Half Way"))
CONGESTION_MAX = "Back to Back"

# Named-workflow endpoint used by the Roboflow inference SDK.
WORKFLOW_URL = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"

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
    for item in data.get("items", []):
        for camera in item.get("cameras", []):
            if camera.get("camera_id") == camera_id:
                return camera.get("image")
    return None


def _run_workflow(image_bytes):
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
        WORKFLOW_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("outputs", [])


def _extract_predictions(outputs):
    """Match the Colab structure, but degrade to an empty list if it differs."""
    try:
        return outputs[0]["predictions"]["predictions"]
    except (IndexError, KeyError, TypeError):
        return []


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


def _congestion_level(y_centres, image_height):
    """Queue depth as the vertical spread of a direction's detections."""
    if not y_centres or not image_height:
        return CONGESTION_BANDS[0][1]
    extent = (max(y_centres) - min(y_centres)) / image_height
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
        },
        "my_sg": {
            "count": counts[DIR_MY_SG],
            "congestion": _congestion_level(centres[DIR_MY_SG], height),
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
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        label = f"{pred.get('class', 'vehicle')} {pred.get('confidence', 0):.2f}"
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=BOX_COLOR)
        draw.text((x1 + 3, ly + 2), label, fill=(0, 0, 0), font=font)

    _draw_banner(draw, image, f"Vehicles: {len(predictions)}")


def _draw_directional(image, predictions, points, summary):
    """Same frame, but boxes coloured by direction and the divider drawn in."""
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 500)
    font = _load_font(max(14, image.width // 90))

    if points:
        draw.line(points, fill=(255, 255, 255), width=max(3, image.width // 384))

    for pred in predictions:
        direction = pred.get("direction", DIR_UNKNOWN)
        color = DIR_COLORS.get(direction, DIR_COLORS[DIR_UNKNOWN])
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{direction} {pred.get('confidence', 0):.2f}"
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    banner = (
        f"{DIR_SG_MY}: {summary['sg_my']['count']} ({summary['sg_my']['congestion']})"
        f"  |  {DIR_MY_SG}: {summary['my_sg']['count']} ({summary['my_sg']['congestion']})"
    )
    _draw_banner(draw, image, banner)


@functions_framework.http
def detect(request):
    if request.method == "OPTIONS":
        return ("", 204, _cors_headers())

    args = request.args or {}
    body = request.get_json(silent=True) or {}

    camera_id = str(body.get("camera_id") or args.get("camera_id") or DEFAULT_CAMERA_ID)
    date_time = body.get("date_time") or args.get("date_time")
    output_format = str(body.get("format") or args.get("format") or "image").lower()
    try:
        min_confidence = float(
            body.get("confidence") or args.get("confidence") or DEFAULT_CONFIDENCE
        )
    except (TypeError, ValueError):
        min_confidence = DEFAULT_CONFIDENCE

    # If no timestamp is supplied, use the current Singapore-local time.
    if not date_time:
        date_time = datetime.datetime.now(ZoneInfo("Asia/Singapore")).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    # 1. Resolve the source image URL from data.gov.sg
    try:
        image_url = _get_camera_image_url(camera_id, date_time)
    except requests.RequestException as exc:
        return _error(f"traffic-images API request failed: {exc}", 502)
    if not image_url:
        return _error(f"No image found for camera {camera_id} at {date_time}", 404)

    # 2. Download the source frame
    try:
        image_bytes = requests.get(image_url, timeout=30).content
    except requests.RequestException as exc:
        return _error(f"Failed to download source image: {exc}", 502)

    # 3. Run the Roboflow workflow over HTTP
    try:
        outputs = _run_workflow(image_bytes)
    except requests.RequestException as exc:
        return _error(f"Inference request failed: {exc}", 502)
    except ValueError as exc:
        return _error(f"Inference returned a non-JSON response: {exc}", 502)

    predictions = _extract_predictions(outputs)
    kept = [p for p in predictions if p.get("confidence", 0) >= min_confidence]

    # 4. Attribute each detection to a direction. Frame size is read from the
    # header alone, so JSON callers still never decode the pixels. Anything that
    # fails here degrades to Unknown rather than failing the request.
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            image_size = probe.size
    except (OSError, ValueError):
        image_size = None
    points = _dividing_line(camera_id, image_size)
    summary = _summarize_directions(kept, points, image_size)

    # Optional JSON mode for inspecting the raw detections
    if output_format == "json":
        payload = {
            "camera_id": camera_id,
            "date_time": date_time,
            "source_image": image_url,
            "min_confidence": min_confidence,
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
        "Cache-Control": "no-store",
    }
    return (out.getvalue(), 200, headers)
