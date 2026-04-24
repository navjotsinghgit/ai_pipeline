"""
product grouping — clusters detected product crops by brand using:
  - HSV color histograms  (dominant packaging color)
  - HOG texture descriptor (packaging texture / layout pattern)
  - Spatial proximity      (products of the same brand are often adjacent)

clustering: dbscan on the combined feature vector.
each cluster → a unique brand group id: GROUP_001, GROUP_002, …
noise points (-1) get their own singleton group.
"""

import colorsys
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import logging
from typing import List

logger = logging.getLogger(__name__)


class ProductGrouper:
    def __init__(self):
    
        self.hog = cv2.HOGDescriptor(
            _winSize=(32, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

        # DBSCAN
        self.dbscan_eps = 0.45
        self.dbscan_min_samples = 1   # every point can be its own cluster

        # colour palette for group labels (BGR)
        np.random.seed(42)
        self._palette = self._gen_palette(64)

    def group(self, image_bgr: np.ndarray,
              detections: List[dict]) -> List[dict]:
        """
        args:
            image_bgr: full original image
            detections: list from detection service
        returns:
            list of {detection_id, brand_group_id, group_color:[B,G,R]}
        """
        if not detections:
            return []

        crops = self._extract_crops(image_bgr, detections)
        features = np.array([self._features(c, d)
                             for c, d in zip(crops, detections)])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="euclidean",
        ).fit_predict(features)

        next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0
        final_labels = []
        for lbl in labels:
            if lbl == -1:
                final_labels.append(next_id)
                next_id += 1
            else:
                final_labels.append(int(lbl))

        results = []
        for det, label in zip(detections, final_labels):
            group_id = f"GROUP_{label + 1:03d}"
            color = self._palette[label % len(self._palette)]
            results.append({
                "detection_id": det["id"],
                "brand_group_id": group_id,
                "group_color": color,  # [B, G, R]
            })

        logger.info(f"Grouped {len(detections)} products into "
                    f"{len(set(final_labels))} brand groups")
        return results

    def _extract_crops(self, image: np.ndarray,
                       detections: List[dict]) -> List[np.ndarray]:
        crops = []
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((64, 32, 3), dtype=np.uint8)
            crops.append(crop)
        return crops

    def _features(self, crop: np.ndarray, det: dict) -> np.ndarray:  
        """combine colour histogram + HOG + spatial features."""

        #  colour histogram in HSV (robust to lighting)
        crop_resized = cv2.resize(crop, (64, 64))
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        colour_feat = np.concatenate([h_hist, s_hist, v_hist])  # 50-dim

        #  hog texture (packaging pattern)
        crop_hog = cv2.resize(crop, (32, 64))
        gray_hog = cv2.cvtColor(crop_hog, cv2.COLOR_BGR2GRAY)
        try:
            hog_out = self.hog.compute(gray_hog)
            if hog_out is None:
                raise ValueError("HOG returned None")
            hog_feat = hog_out.flatten()
        except Exception:
            hog_feat = np.zeros(756)

        #  spatial position (normalised bbox centre)
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # weight spatial features less than colour/texture
        spatial = np.array([cx / 2000.0, cy / 2000.0]) * 0.3

        return np.concatenate([colour_feat, hog_feat, spatial])
    @staticmethod
    def _gen_palette(n: int) -> List[list]:
        """generate n visually distinct BGR colours using golden-ratio HSV."""
        colours = []
        golden = 0.618033988749895
        h = 0.0
        for _ in range(n):
            h = (h + golden) % 1.0
            rgb = np.array(
                [int(x * 255) for x in _hsv_to_rgb(h, 0.75, 0.90)]
            )
            colours.append([int(rgb[2]), int(rgb[1]), int(rgb[0])])  # BGR
        return colours


def _hsv_to_rgb(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)
