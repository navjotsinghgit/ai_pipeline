"""
grouping microservice — port 5002
accepts image + detections, returns brand group assignments.
"""

import base64
import logging
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

from grouper import ProductGrouper

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [grouping] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
grouper = ProductGrouper()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "grouping"})


@app.route("/group", methods=["POST"])
def group():
    data = request.get_json(force=True)
    request_id = data.get("request_id", "unknown")

    if "image_b64" not in data or "detections" not in data:
        return jsonify({"error": "image_b64 and detections fields required"}), 400

    t0 = time.perf_counter()

    try:
        img_bytes = base64.b64decode(data["image_b64"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return jsonify({"error": str(e)}), 400

    detections = data["detections"]
    groups = grouper.group(image, detections)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    response = {
        "request_id": request_id,
        "groups": groups,
        "grouping_time_ms": elapsed_ms,
    }
    logger.info(f"[{request_id}] Grouping done in {elapsed_ms}ms")
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
