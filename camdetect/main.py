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

# Named-workflow endpoint used by the Roboflow inference SDK.
WORKFLOW_URL = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
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


def _load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


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

    banner = f"Vehicles: {len(predictions)}"
    bfont = _load_font(max(18, image.width // 60))
    box = draw.textbbox((0, 0), banner, font=bfont)
    draw.rectangle([0, 0, box[2] - box[0] + 16, box[3] - box[1] + 12], fill=(0, 0, 0))
    draw.text((8, 6), banner, fill=(255, 255, 255), font=bfont)


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

    # Optional JSON mode for inspecting the raw detections
    if output_format == "json":
        payload = {
            "camera_id": camera_id,
            "date_time": date_time,
            "source_image": image_url,
            "min_confidence": min_confidence,
            "vehicle_count": len(kept),
            "predictions": kept,
        }
        headers = {**_cors_headers(), "Content-Type": "application/json"}
        return (json.dumps(payload), 200, headers)

    # 4. Draw the boxes on the full-resolution frame and return it
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    _draw_boxes(image, kept)

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    headers = {
        **_cors_headers(),
        "Content-Type": "image/jpeg",
        "X-Vehicle-Count": str(len(kept)),
        "X-Source-Image": image_url,
        "X-Frame-Datetime": date_time,
        "Cache-Control": "no-store",
    }
    return (out.getvalue(), 200, headers)
