"""
severity_estimator.py
--------------------
Disease Severity Estimator for ArecaVision AI.

Calculates the percentage of affected tissue on diseased leaf, fruit, or trunk surfaces
using HSV color space segmentation (identifying necrosis, yellowing, water-soaked lesions,
or rot) combined with Grad-CAM class activation intensity weighting.

Severity Classification:
  - Healthy (0%)
  - Mild Infection (< 15% surface affected)
  - Moderate Infection (15% - 40% surface affected)
  - Severe Infection (> 40% surface affected)
"""

import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger("SeverityEstimator")


def estimate_disease_severity(image_np: np.ndarray, predicted_class: str,
                              heatmap_2d: np.ndarray = None) -> dict:
    """
    Estimates the percentage of affected plant surface and determines the severity tier.

    Args:
        image_np: RGB image array of shape (224, 224, 3) in [0, 255] range float32 or uint8.
        predicted_class: Name of the predicted condition.
        heatmap_2d: Optional Grad-CAM 2D heatmap array normalized to [0, 1].

    Returns:
        dict containing:
            - severity_pct: float (0.0 to 100.0)
            - severity_level: str ("Healthy", "Mild", "Moderate", "Severe")
            - status_color: str (Hex color for UI display)
            - affected_area_pixels: int
            - total_plant_pixels: int
    """
    # All class names that represent healthy palm states (no disease lesion analysis needed)
    healthy_classes = frozenset({"healthy_foot", "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk"})

    if predicted_class in healthy_classes:
        return {
            "severity_pct": 0.0,
            "severity_level": "Healthy",
            "status_color": "#2E7559",
            "affected_area_pixels": 0,
            "total_plant_pixels": 224 * 224,
            "description": "Baseline health state — no disease lesions or surface decay detected."
        }

    # Convert to uint8 RGB
    if image_np.max() <= 1.0:
        img_uint8 = np.uint8(255 * image_np)
    else:
        img_uint8 = np.uint8(image_np)

    # Convert RGB to HSV color space
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Define color thresholds for disease symptoms in Arecanut palms
    # 1. Yellow Leaf Disease / Yellowing (Hue: 15–35)
    yellow_lower = np.array([15, 40, 40])
    yellow_upper = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 2. Brown/Necrotic/Rot Lesions (Mahali, Koleroga, Stem Bleeding, Bud Borer) (Hue: 0–15 & 160–180, low value)
    brown_lower1 = np.array([0, 30, 20])
    brown_upper1 = np.array([18, 255, 180])
    brown_lower2 = np.array([160, 30, 20])
    brown_upper2 = np.array([180, 255, 180])
    mask_brown = cv2.inRange(hsv, brown_lower1, brown_upper1) | cv2.inRange(hsv, brown_lower2, brown_upper2)

    # Combine disease lesion mask
    lesion_mask = cv2.bitwise_or(mask_yellow, mask_brown)

    # If Grad-CAM heatmap is provided, weight lesions by neural attention focus (>0.3 threshold)
    if heatmap_2d is not None and heatmap_2d.shape == (224, 224):
        attention_mask = np.uint8(heatmap_2d > 0.3) * 255
        combined_mask = cv2.bitwise_and(lesion_mask, attention_mask)
        # Fallback to lesion mask if combined mask is too sparse
        if np.sum(combined_mask > 0) < 50:
            final_mask = lesion_mask
        else:
            final_mask = combined_mask
    else:
        final_mask = lesion_mask

    # Calculate plant foreground mask (non-black pixels)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, foreground_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    total_foreground_pixels = max(1, np.sum(foreground_mask > 0))

    affected_pixels = np.sum((final_mask > 0) & (foreground_mask > 0))
    severity_pct = float(min(100.0, (affected_pixels / total_foreground_pixels) * 100.0 * 1.5))

    # Categorize into severity levels
    if severity_pct < 15.0:
        level = "Mild"
        color = "#F1C40F"  # Yellow
        desc = "Early stage infection (<15% surface area affected). Organic treatment highly effective."
    elif severity_pct < 40.0:
        level = "Moderate"
        color = "#E67E22"  # Orange
        desc = "Moderate infection (15%–40% surface area affected). Timely chemical intervention recommended."
    else:
        level = "Severe"
        color = "#E74C3C"  # Red
        desc = "Critical infection (>40% surface area affected). Immediate containment required to prevent field outbreak."

    logger.info(f"Severity estimate for {predicted_class}: {severity_pct:.1f}% ({level})")

    return {
        "severity_pct": round(severity_pct, 1),
        "severity_level": level,
        "status_color": color,
        "affected_area_pixels": int(affected_pixels),
        "total_plant_pixels": int(total_foreground_pixels),
        "description": desc
    }
