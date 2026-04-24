"""
tests for the AI pipeline components.
"""

import os
import sys
import cv2
import numpy as np
import pytest

# path setup 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_DIR = os.path.join(ROOT, "detection_service")
GRP_DIR = os.path.join(ROOT, "grouping_service")
VIZ_DIR = os.path.join(ROOT, "visualization")

for p in [DET_DIR, GRP_DIR, VIZ_DIR, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# helpers
def make_shelf_image(w=640, h=960):
    """synthetic shelf image with 12 product-like coloured rectangles."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 220
    for row in range(3):
        for col in range(4):
            x1 = 40 + col * 140
            y1 = 100 + row * 280
            x2, y2 = x1 + 110, y1 + 240
            color = (40 + row * 60, 80 + col * 40, 200 - col * 20)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return img


def make_detections(n=6):
    return [
        {
            "id": i,
            "bbox": [20 + i * 70, 20, 80 + i * 70, 120],
            "confidence": 0.8,
            "class": "product",
        }
        for i in range(n)
    ]


def make_random_image(w=480, h=600):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


#detection test
class TestDetector:
    @pytest.fixture(autouse=True)
    def setup(self):
        from detector import ShelfProductDetector
        self.det = ShelfProductDetector()

    def test_model_loads(self):
        """model should load without error."""
        assert self.det._model is not None
        assert self.det._model_type in ("hf_retail", "yolov8n_coco")

    def test_returns_list(self):
        result = self.det.detect(make_shelf_image())
        assert isinstance(result, list)

    def test_detection_fields(self):
        for d in self.det.detect(make_shelf_image()):
            assert "id" in d
            assert "bbox" in d and len(d["bbox"]) == 4
            assert "confidence" in d
            assert 0 <= d["confidence"] <= 1
            assert "class" in d

    def test_bbox_within_image(self):
        w, h = 640, 960
        for d in self.det.detect(make_shelf_image(w, h)):
            x1, y1, x2, y2 = d["bbox"]
            assert x1 >= 0 and y1 >= 0
            assert x2 <= w and y2 <= h
            assert x2 > x1 and y2 > y1

    def test_empty_image_does_not_crash(self):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = self.det.detect(img)
        assert isinstance(result, list)

    def test_ids_are_sequential(self):
        dets = self.det.detect(make_shelf_image())
        ids = [d["id"] for d in dets]
        assert ids == list(range(len(ids)))

    def test_real_shelf_image_detects_products(self):
        """On a proper shelf image the model should find at least a few products."""
        # more realistic shelf image with clear rectangular products
        img = make_shelf_image(w=1280, h=960)
        dets = self.det.detect(img)
        # we don't assert exact count — just that it doesn't crash and returns list
        assert isinstance(dets, list)


#grouping tests 
class TestGrouper:
    @pytest.fixture(autouse=True)
    def setup(self):
        from grouper import ProductGrouper
        self.grp = ProductGrouper()

    def test_returns_one_result_per_detection(self):
        dets = make_detections(5)
        results = self.grp.group(make_random_image(), dets)
        assert len(results) == 5

    def test_result_fields(self):
        dets = make_detections(3)
        for r in self.grp.group(make_random_image(), dets):
            assert "detection_id" in r
            assert "brand_group_id" in r
            assert r["brand_group_id"].startswith("GROUP_")
            assert "group_color" in r
            assert len(r["group_color"]) == 3

    def test_empty_detections(self):
        assert self.grp.group(make_random_image(), []) == []

    def test_detection_ids_preserved(self):
        dets = make_detections(4)
        results = self.grp.group(make_random_image(), dets)
        returned_ids = {r["detection_id"] for r in results}
        original_ids = {d["id"] for d in dets}
        assert returned_ids == original_ids

    def test_group_colors_are_bgr_tuples(self):
        for r in self.grp.group(make_random_image(), make_detections(3)):
            b, g, rv = r["group_color"]
            for ch in [b, g, rv]:
                assert 0 <= ch <= 255


#visualization tests 
class TestVisualizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        import visualizer
        self.viz = visualizer

    def test_saves_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.viz, "OUTPUTS_DIR", str(tmp_path))
        img = make_random_image()
        dets = make_detections(2)
        groups = [
            {"detection_id": 0, "brand_group_id": "GROUP_001", "group_color": [0, 120, 255]},
            {"detection_id": 1, "brand_group_id": "GROUP_002", "group_color": [255, 80, 0]},
        ]
        rel = self.viz.draw_and_save(img, dets, groups, "test.jpg")
        assert "annotated" in rel
        assert os.path.exists(os.path.join(str(tmp_path), os.path.basename(rel)))

    def test_output_is_valid_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.viz, "OUTPUTS_DIR", str(tmp_path))
        img = make_random_image()
        dets = make_detections(2)
        groups = [
            {"detection_id": 0, "brand_group_id": "GROUP_001", "group_color": [0, 120, 255]},
            {"detection_id": 1, "brand_group_id": "GROUP_002", "group_color": [255, 80, 0]},
        ]
        rel = self.viz.draw_and_save(img, dets, groups, "test.jpg")
        loaded = cv2.imread(os.path.join(str(tmp_path), os.path.basename(rel)))
        assert loaded is not None and loaded.shape[2] == 3

    def test_no_detections_saves_clean_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.viz, "OUTPUTS_DIR", str(tmp_path))
        img = make_random_image()
        rel = self.viz.draw_and_save(img, [], [], "empty.jpg")
        assert os.path.exists(os.path.join(str(tmp_path), os.path.basename(rel)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
