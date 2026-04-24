"""
visualizing module draws color-coded bounding boxes for brand groups,
saves annotated images to the outputs/ directory.
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import List

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def draw_and_save(
    image_bgr: np.ndarray,
    detections: List[dict],
    groups: List[dict],
    original_filename: str,
) -> str:
    """
    draw bounding boxes colour-coded by brand group on image.
    saves file to outputs/ and returns the relative path.
    """

    group_map = {g["detection_id"]: g for g in groups}

    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    scale = max(w, h) / 1000.0
    thickness = max(2, int(2 * scale))
    font_scale = max(0.4, 0.5 * scale)

    legend_entries = {}

    for det in detections:
        det_id = det["id"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 0)

        group_info = group_map.get(det_id, {})
        group_id = group_info.get("brand_group_id", "UNGROUPED")
        color = group_info.get("group_color", [128, 128, 128])
        color_tuple = tuple(int(c) for c in color)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_tuple, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_tuple, thickness)

        label = f"{group_id} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, 1)
        lx1, ly1 = x1, max(y1 - th - baseline - 4, 0)
        lx2, ly2 = x1 + tw + 4, y1
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color_tuple, -1)

        cv2.putText(annotated, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)

        legend_entries[group_id] = color_tuple


    annotated = _draw_legend(annotated, legend_entries, scale)


    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(original_filename)[0]
    out_name = f"{stem}_annotated_{timestamp}.jpg"
    out_path = os.path.join(OUTPUTS_DIR, out_name)
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    logger.info(f"Saved visualization → {out_path}")
    return f"outputs/{out_name}"


def _draw_legend(img: np.ndarray, entries: dict, scale: float) -> np.ndarray:
    if not entries:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, 0.45 * scale)
    pad = 8
    swatch = int(16 * scale)
    line_h = swatch + 6

    panel_w = int(160 * scale)
    panel_h = pad * 2 + len(entries) * line_h + 20
    h, w = img.shape[:2]

    px = w - panel_w - pad
    py = h - panel_h - pad


    overlay = img.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Title
    cv2.putText(img, "Brand Groups", (px + pad, py + pad + 12),
                font, fs, (200, 200, 200), 1, cv2.LINE_AA)

    for i, (gid, color) in enumerate(sorted(entries.items())):
        y = py + pad + 24 + i * line_h
        cv2.rectangle(img, (px + pad, y), (px + pad + swatch, y + swatch),
                      color, -1)
        cv2.putText(img, gid, (px + pad + swatch + 6, y + swatch - 3),
                    font, fs * 0.85, (220, 220, 220), 1, cv2.LINE_AA)

    return img
