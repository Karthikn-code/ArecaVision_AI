"""
test_object_detection.py
-------------------------
Unit tests for ArecaVision AI Object Detection package:
- Annotations parser
- OpenCV bounding box overlay renderer
- LeafDiseaseDetector prediction engine
"""

import os
import unittest
import numpy as np
import cv2
import pandas as pd

from object_detection.dataset_parser import parse_annotations_csv, summarize_annotations
from object_detection.overlay import render_bounding_boxes, draw_detection_legend, get_color_for_class
from object_detection.detector import LeafDiseaseDetector, compute_iou, non_max_suppression


class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.test_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
        # Draw a simulated yellow disease spot
        cv2.circle(self.test_img, (200, 200), 40, (10, 220, 220), -1)

    def test_compute_iou(self):
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)

        box3 = [100, 100, 150, 150]
        self.assertEqual(compute_iou(box1, box3), 0.0)

    def test_non_max_suppression(self):
        boxes = [
            {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "confidence": 0.9},
            {"xmin": 12, "ymin": 12, "xmax": 52, "ymax": 52, "confidence": 0.8}, # Overlapping
            {"xmin": 200, "ymin": 200, "xmax": 250, "ymax": 250, "confidence": 0.95}
        ]
        suppressed = non_max_suppression(boxes, iou_threshold=0.4)
        self.assertEqual(len(suppressed), 2)
        self.assertEqual(suppressed[0]["confidence"], 0.95)
        self.assertEqual(suppressed[1]["confidence"], 0.9)

    def test_overlay_rendering(self):
        boxes = [
            {"xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150, "class": "DiseaseSpot", "confidence": 0.85},
            {"xmin": 200, "ymin": 200, "xmax": 300, "ymax": 300, "class": "non-diseased-leaf", "confidence": 0.92}
        ]
        annotated = render_bounding_boxes(self.test_img, boxes)
        self.assertEqual(annotated.shape, self.test_img.shape)
        self.assertFalse(np.array_equal(annotated, self.test_img))

        counts = {"DiseaseSpot": 1, "non-diseased-leaf": 1}
        legend_img = draw_detection_legend(annotated, counts)
        self.assertEqual(legend_img.shape, self.test_img.shape)

    def test_leaf_disease_detector_prediction(self):
        detector = LeafDiseaseDetector(conf_threshold=0.30)
        res = detector.predict(self.test_img, filename="test_spot.jpg")

        self.assertIn("boxes", res)
        self.assertIn("disease_spots_count", res)
        self.assertIn("annotated_image", res)
        self.assertEqual(res["annotated_image"].shape, (400, 400, 3))
        self.assertIsInstance(res["disease_spots_count"], int)

    def test_color_class_mapping(self):
        col_disease = get_color_for_class("DiseaseSpot")
        col_healthy = get_color_for_class("non-diseased-leaf")
        self.assertEqual(col_disease, (0, 0, 220))
        self.assertEqual(col_healthy, (30, 180, 50))


if __name__ == "__main__":
    unittest.main()
