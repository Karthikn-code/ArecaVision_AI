"""
Object Detection Package for ArecaVision AI.
Provides annotations parser, bounding box detector, and OpenCV visual overlay renderer.
"""

from .dataset_parser import parse_annotations_csv, get_dataset_bounding_boxes
from .overlay import render_bounding_boxes, draw_detection_legend
from .detector import LeafDiseaseDetector

__all__ = [
    "parse_annotations_csv",
    "get_dataset_bounding_boxes",
    "render_bounding_boxes",
    "draw_detection_legend",
    "LeafDiseaseDetector",
]
