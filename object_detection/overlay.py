"""
overlay.py
----------
OpenCV visualizer for rendering color-coded bounding boxes, class labels,
and confidence badges onto field images for Areca leaf disease diagnosis.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Color Palette (BGR format for OpenCV)
COLOR_DISEASE_SPOT = (0, 0, 220)       # Vivid Red
COLOR_DISEASE_LEAF = (0, 100, 255)     # Deep Orange
COLOR_HEALTHY_LEAF = (30, 180, 50)     # Emerald Green
COLOR_DEFAULT = (255, 140, 0)          # Amber / Blue-Orange

CLASS_COLOR_MAP = {
    "disease-leaf": COLOR_DISEASE_LEAF,
    "Disease": COLOR_DISEASE_SPOT,
    "DiseaseSpot": COLOR_DISEASE_SPOT,
    "non-diseased-leaf": COLOR_HEALTHY_LEAF,
    "NonDisease": COLOR_HEALTHY_LEAF,
    "HealthySpot": COLOR_HEALTHY_LEAF,
}


def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color tuple for a given class label."""
    return CLASS_COLOR_MAP.get(class_name, COLOR_DEFAULT)


def render_bounding_boxes(
    image: np.ndarray,
    boxes: List[Dict[str, Any]],
    line_thickness: int = 2,
    font_scale: float = 0.5,
    draw_badge: bool = True
) -> np.ndarray:
    """
    Render bounding boxes with label badges on an input OpenCV image (RGB or BGR).
    Input image should be uint8 numpy array. Returns annotated copy of image.

    Each box dict expects:
    - xmin, ymin, xmax, ymax (int pixel coordinates)
    - class (str)
    - confidence (float, 0.0 to 1.0, optional)
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for box in boxes:
        label = box.get("class", "Detection")
        confidence = box.get("confidence", 1.0)
        xmin = max(0, int(box.get("xmin", 0)))
        ymin = max(0, int(box.get("ymin", 0)))
        xmax = min(img_w, int(box.get("xmax", img_w)))
        ymax = min(img_h, int(box.get("ymax", img_h)))

        color = get_color_for_class(label)

        # Draw main bounding rectangle
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, line_thickness)

        if draw_badge:
            # Format text badge
            conf_str = f"{confidence * 100:.0f}%" if confidence < 1.0 else ""
            badge_text = f"{label} {conf_str}".strip()

            # Calculate text size for badge background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, 1)

            # Draw filled rectangle for badge background above box
            badge_ymin = max(0, ymin - text_h - 6)
            badge_ymax = ymin
            badge_xmax = min(img_w, xmin + text_w + 8)

            cv2.rectangle(annotated, (xmin, badge_ymin), (badge_xmax, badge_ymax), color, -1)

            # Write text on badge
            cv2.putText(
                annotated,
                badge_text,
                (xmin + 4, ymin - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return annotated


def draw_detection_legend(
    image: np.ndarray,
    counts: Dict[str, int]
) -> np.ndarray:
    """
    Draw a summary legend overlay box on the top-left corner of the image.
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Legend dimensions
    padding = 10
    box_w = 230
    box_h = 24 + (len(counts) * 20)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    # Blend semi-transparent background
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, "Detections Summary", (18, 28), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    y_offset = 48
    for label, count in counts.items():
        color = get_color_for_class(label)
        # Small color swatch
        cv2.rectangle(annotated, (18, y_offset - 10), (30, y_offset), color, -1)
        cv2.putText(annotated, f"{label}: {count}", (38, y_offset), font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y_offset += 20

    return annotated
