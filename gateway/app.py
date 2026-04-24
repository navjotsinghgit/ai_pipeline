"""
flask API gateway port 5000
full AI pipeline:
  1. accept image from browser
  2. send to detection service (5001)
  3. send image & detections to grouping service (5002)
  4. merge results, run visualization, return JSON + annotated image
"""

import base64
import logging
import os
import sys
import time
import uuid
from typing import List

import cv2
import numpy as np
import requests
from flask import (Flask, jsonify, render_template, request,
                   send_from_directory)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visualization.visualizer import draw_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gateway] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DETECTION_URL = os.environ.get("DETECTION_URL", "http://localhost:5001")
GROUPING_URL  = os.environ.get("GROUPING_URL",  "http://localhost:5002")

TIMEOUT = 60          
MAX_IMAGE_BYTES = 20 * 1024 * 1024   


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    services = {}
    for name, url in [("detection", DETECTION_URL), ("grouping", GROUPING_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            services[name] = r.json()
        except Exception as e:
            services[name] = {"status": "unreachable", "error": str(e)}
    return jsonify({"status": "ok", "service": "gateway", "upstream": services})


@app.route("/analyze", methods=["POST"])
def analyze():
    t_start = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field: 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    raw = file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large (max 20 MB)"}), 413

    request_id = str(uuid.uuid4())
    filename = file.filename or "upload.jpg"
    logger.info(f"[{request_id}] received {filename} ({len(raw)//1024} KB)")

    image_b64 = base64.b64encode(raw).decode("utf-8")

    try:
        det_resp = requests.post(
            f"{DETECTION_URL}/detect",
            json={"request_id": request_id, "image_b64": image_b64},
            timeout=TIMEOUT,
        )
        det_resp.raise_for_status()
        det_data = det_resp.json()
    except Exception as e:
        logger.error(f"Detection service error: {e}")
        return jsonify({"error": f"Detection service unavailable: {e}"}), 503

    detections   = det_data.get("detections", [])
    image_shape  = det_data.get("image_shape", [0, 0])
    det_time_ms  = det_data.get("detection_time_ms", 0)
    model_used   = det_data.get("model_used", "unknown")

    try:
        grp_resp = requests.post(
            f"{GROUPING_URL}/group",
            json={
                "request_id": request_id,
                "image_b64": image_b64,
                "detections": detections,
            },
            timeout=TIMEOUT,
        )
        grp_resp.raise_for_status()
        grp_data = grp_resp.json()
    except Exception as e:
        logger.error(f"Grouping service error: {e}")
        return jsonify({"error": f"Grouping service unavailable: {e}"}), 503

    groups      = grp_data.get("groups", [])
    grp_time_ms = grp_data.get("grouping_time_ms", 0)

    # merge results
    group_map = {g["detection_id"]: g for g in groups}
    merged = []
    brand_summary = {}

    for det in detections:
        grp = group_map.get(det["id"], {})
        gid   = grp.get("brand_group_id", "GROUP_000")
        color = grp.get("group_color", [128, 128, 128])
        merged.append({**det, "brand_group_id": gid, "group_color": color})

        if gid not in brand_summary:
            brand_summary[gid] = {"count": 0, "color": color}
        brand_summary[gid]["count"] += 1

    nparr  = np.frombuffer(raw, np.uint8)
    image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    vis_path = draw_and_save(image, detections, groups, filename)

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(f"[{request_id}] pipeline done in {total_ms}ms | "
                f"detected={len(detections)} groups={len(brand_summary)}")

    return jsonify({
        "request_id":              request_id,
        "status":                  "success",
        "processing_time_ms":      total_ms,
        "detection_time_ms":       det_time_ms,
        "grouping_time_ms":        grp_time_ms,
        "model_used":              model_used,
        "image_width":             image_shape[1] if len(image_shape) > 1 else 0,
        "image_height":            image_shape[0] if len(image_shape) > 0 else 0,
        "total_products_detected": len(detections),
        "total_brand_groups":      len(brand_summary),
        "detections":              merged,
        "brand_groups":            brand_summary,
        "visualization_url":       f"/{vis_path}",
    })

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(os.path.abspath(OUTPUTS_DIR), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
