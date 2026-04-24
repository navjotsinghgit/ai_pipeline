"""
detection microservice — port 5001
accepts image, returns product bounding boxes via YOLOv8.
"""

import base64
import logging
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

from detector import ShelfProductDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [detection] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Initializing detector (model download may take a moment)…")
detector = ShelfProductDetector()
logger.info("Detector ready.")


@app.route("/health", methods=["GET"])
def health():
    model_type = detector._model_type or "unavailable"
    return jsonify({
        "status":     "ok",
        "service":    "detection",
        "model":      model_type,
        "model_ready": detector._model is not None,
    })


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    request_id = data.get("request_id", "unknown")

    if "image_b64" not in data:
        return jsonify({"error": "image_b64 field required"}), 400

    t0 = time.perf_counter()

    try:
        img_bytes = base64.b64decode(data["image_b64"])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return jsonify({"error": str(e)}), 400

    detections  = detector.detect(image)
    elapsed_ms  = round((time.perf_counter() - t0) * 1000, 2)

    response = {
        "request_id":        request_id,
        "detections":        detections,
        "image_shape":       [image.shape[0], image.shape[1]],
        "detection_time_ms": elapsed_ms,
        "model_used":        detector._model_type,
    }
    logger.info(
        f"[{request_id}] {len(detections)} detections | {elapsed_ms}ms"
    )
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
