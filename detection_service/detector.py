import logging
import math
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_OPTIONS = [
    ("keremberke/yolov8m-retail-shelf-detection", "hf_retail_medium"),
    ("keremberke/yolov8n-retail-shelf-detection", "hf_retail_nano"),
    ("yolov8m.pt", "yolov8m_coco"),
    ("yolov8n.pt", "yolov8n_coco"),
]

COCO_PRODUCT_CLASSES = {39, 41, 42, 43, 44, 45, 63, 67, 73, 74, 75, 76, 79}

CONF_THRESH     = 0.20    
IOU_THRESH      = 0.40    
GLOBAL_NMS_IOU  = 0.45    
MAX_DET         = 500
TILE_SIZE       = 640     
TILE_OVERLAP    = 0.25    
MIN_DIM_TO_TILE = 800     


class ShelfProductDetector:
    def __init__(self):
        self._model      = None
        self._model_type = None
        self._load_model()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load_model(self):
        from ultralytics import YOLO

        for model_id, label in MODEL_OPTIONS:
            try:
                logger.info(f"Trying model: {model_id}")
                m = YOLO(model_id)
                _ = m.names            
                self._model      = m
                self._model_type = label
                logger.info(f"Model loaded: {label} ✓")
                return
            except Exception as e:
                logger.warning(f"  ✗ {model_id}: {e}")

        logger.error("All models failed to load.")

    def detect(self, image_bgr: np.ndarray) -> List[dict]:
        if self._model is None:
            logger.error("No model — returning empty")
            return []

        orig_h, orig_w = image_bgr.shape[:2]

        preprocessed = self._clahe_enhance(image_bgr)

        use_tiling = (orig_h > MIN_DIM_TO_TILE or orig_w > MIN_DIM_TO_TILE)

        if use_tiling:
            raw_boxes, raw_scores, raw_classes = self._tiled_predict(
                preprocessed, orig_w, orig_h
            )
        else:
            raw_boxes, raw_scores, raw_classes = self._single_predict(
                preprocessed
            )

        kept = self._nms(raw_boxes, raw_scores, GLOBAL_NMS_IOU)

        names = self._model.names
        detections = []
        for idx, i in enumerate(kept):
            x1, y1, x2, y2 = [int(v) for v in raw_boxes[i]]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(orig_w, x2); y2 = min(orig_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "id":         idx,
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(float(raw_scores[i]), 3),
                "class":      names.get(int(raw_classes[i]), "product"),
            })

        logger.info(
            f"[{self._model_type}] {len(detections)} products detected"
            f"{' (tiled)' if use_tiling else ''}"
        )
        return detections

    def _single_predict(
        self, image: np.ndarray
    ) -> Tuple[List, List, List]:
        """runs YOLO on full image and return raw (boxes, scores, classes)."""
        results = self._model.predict(
            source=image,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            max_det=MAX_DET,
            augment=True,          
            verbose=False,
            device="cpu",
        )
        return self._parse_results(results)

    def _tiled_predict(
        self, image: np.ndarray, orig_w: int, orig_h: int
    ) -> Tuple[List, List, List]:
        """
        slices image into overlapping tiles, runs YOLO on each,
        maps detections back to original coordinates.
        """
        stride = int(TILE_SIZE * (1 - TILE_OVERLAP))

        boxes, scores, classes = [], [], []

        # Compute tile grid
        x_starts = list(range(0, max(orig_w - TILE_SIZE, 1), stride))
        if not x_starts or x_starts[-1] + TILE_SIZE < orig_w:
            x_starts.append(max(0, orig_w - TILE_SIZE))

        y_starts = list(range(0, max(orig_h - TILE_SIZE, 1), stride))
        if not y_starts or y_starts[-1] + TILE_SIZE < orig_h:
            y_starts.append(max(0, orig_h - TILE_SIZE))

        for ys in y_starts:
            for xs in x_starts:
                ye = min(ys + TILE_SIZE, orig_h)
                xe = min(xs + TILE_SIZE, orig_w)
                tile = image[ys:ye, xs:xe]

                # pad to TILE_SIZE × TILE_SIZE (bottom-right padding)
                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_h, 0, pad_w,
                        cv2.BORDER_CONSTANT, value=(114, 114, 114)
                    )

                results = self._model.predict(
                    source=tile,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    max_det=MAX_DET,
                    augment=False,   
                    verbose=False,
                    device="cpu",
                )
                tb, ts, tc = self._parse_results(results)

                # filter COCO fallback classes
                for b, s, c in zip(tb, ts, tc):
                    if (self._model_type == "yolov8m_coco"
                            and int(c) not in COCO_PRODUCT_CLASSES):
                        continue
                    # map tile-relative coords to original image coords
                    tx1, ty1, tx2, ty2 = b
                    boxes.append([tx1 + xs, ty1 + ys, tx2 + xs, ty2 + ys])
                    scores.append(s)
                    classes.append(c)

        return boxes, scores, classes

    @staticmethod
    def _parse_results(results) -> Tuple[List, List, List]:
        boxes, scores, classes = [], [], []
        if not results:
            return boxes, scores, classes
        bx = results[0].boxes
        if bx is None or len(bx) == 0:
            return boxes, scores, classes
        xyxy    = bx.xyxy.cpu().numpy()
        confs   = bx.conf.cpu().numpy()
        cls_ids = bx.cls.cpu().numpy()
        for i in range(len(xyxy)):
            boxes.append(xyxy[i].tolist())
            scores.append(float(confs[i]))
            classes.append(float(cls_ids[i]))
        return boxes, scores, classes

    def _clahe_enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB to fix uneven shelf lighting."""
        try:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self._clahe.apply(l)
            enhanced = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            return image_bgr   

    @staticmethod
    def _nms(boxes: List, scores: List, iou_thresh: float) -> List[int]:
        if not boxes:
            return []
        arr    = np.array(boxes,  dtype=np.float32)
        s_arr  = np.array(scores, dtype=np.float32)
        order  = s_arr.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(arr[i, 0], arr[order[1:], 0])
            yy1 = np.maximum(arr[i, 1], arr[order[1:], 1])
            xx2 = np.minimum(arr[i, 2], arr[order[1:], 2])
            yy2 = np.minimum(arr[i, 3], arr[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (arr[i, 2] - arr[i, 0]) * (arr[i, 3] - arr[i, 1])
            area_j = ((arr[order[1:], 2] - arr[order[1:], 0]) *
                      (arr[order[1:], 3] - arr[order[1:], 1]))
            iou    = inter / (area_i + area_j - inter + 1e-6)
            order  = order[1:][iou < iou_thresh]
        return keep
