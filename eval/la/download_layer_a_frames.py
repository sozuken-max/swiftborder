#!/usr/bin/env python3
"""Download the 15 GCS originals listed in ``runs/layer_a_count/artifact_counts.json``.

Writes each object basename into the output directory (1920 x 1080 JPEGs). Requires Application Default Credentials with read access to ``sg-lta-traffic-cameras``
(``pip install google-cloud-storage``), or copy with ``gsutil`` as documented in
``runs/layer_a_count/REPORT.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from google.cloud import storage
except ImportError:
    storage = None  # type: ignore[assignment]

from layer_a_count_lib import RUN_DIR, gcs_object_name, load_artifact

DEFAULT_OUT = RUN_DIR / "frames"


def main() -> None:
    if storage is None:
        raise SystemExit("google-cloud-storage is not installed; use gsutil (see runs/layer_a_count/REPORT.md)")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    artifact = load_artifact()
    for image in artifact["images"]:
        gcs_path = image["gcs_path"]
        assert gcs_path.startswith("gs://"), gcs_path
        bucket_name, object_name = gcs_path[5:].split("/", 1)
        dest = out / gcs_object_name(gcs_path)
        if dest.exists() and dest.stat().st_size > 0:
            print("skip", dest.name)
            continue
        blob = client.bucket(bucket_name).blob(object_name)
        blob.download_to_filename(dest)
        print("ok", dest.name, dest.stat().st_size)
    print("done", out)


if __name__ == "__main__":
    main()
