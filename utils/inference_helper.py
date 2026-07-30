"""
inference_helper.py
-------------------
Centralized single-image inference pipeline shared across all pages.
Handles preprocessing → model predict → Grad-CAM → DB save.
"""

import os
import uuid
import time
import numpy as np
import cv2
from PIL import Image

from config.config import RESULTS_DIR, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
from augmentation.augmentor import preprocess_and_denoise_image
from gradcam.gradcam import compute_gradcam, overlay_heatmap
from models.model_registry import load_or_create_model, ensemble_predict
from database.db_manager import save_prediction
from utils.logger import get_logger

logger = get_logger("InferenceHelper")


def validate_image(image_path: str) -> bool:
    """Checks whether an image file is valid and non-corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def run_inference(image_path: str, model_name: str, use_ensemble: bool = False) -> dict:
    """
    End-to-end inference pipeline for a single image file.

    Args:
        image_path: Absolute path to the image on disk.
        model_name: Architecture name (e.g. 'EfficientNet-B0').
        use_ensemble: If True, runs soft-voting ensemble across all 3 models.

    Returns:
        result dict containing:
            - prediction_id: str
            - predicted_class: str
            - display_name: str
            - confidence: float
            - processing_time: float
            - model_used: str
            - predictions_raw: np.ndarray (shape: [num_classes])
            - gradcam_path: str or None
            - preprocessed_img: np.ndarray (float32, [0-255])
    """
    from config.config import DISPLAY_NAMES

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    start_time = time.time()

    # 1. Preprocess (denoise + resize)
    preprocessed_img = preprocess_and_denoise_image(image_path)
    img_batch = np.expand_dims(preprocessed_img, axis=0)   # (1, 224, 224, 3)

    # 2. Model inference
    gradcam_model = None
    if use_ensemble:
        predictions, ensemble_models = ensemble_predict(img_batch)
        gradcam_model = ensemble_models.get("EfficientNet-B0")
        used_model_label = "Ensemble (EfficientNet-B0 + MobileNetV3 + ResNet50)"
    else:
        gradcam_model = load_or_create_model(model_name)
        predictions = gradcam_model.predict(img_batch, verbose=0)
        used_model_label = model_name

    end_time = time.time()
    processing_time = end_time - start_time

    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    predicted_class = CLASS_NAMES[pred_idx]
    display_name = DISPLAY_NAMES.get(predicted_class, predicted_class)

    logger.info(
        f"Inference complete — class='{predicted_class}', confidence={confidence:.3f}, "
        f"time={processing_time:.3f}s, model={used_model_label}"
    )

    # 3. Grad-CAM
    gradcam_path = None
    temp_dir = os.path.join(RESULTS_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    if gradcam_model is not None:
        try:
            heatmap = compute_gradcam(gradcam_model, img_batch, class_index=pred_idx)
            overlay = overlay_heatmap(heatmap, preprocessed_img, alpha=0.5)
            gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
            gradcam_path = os.path.join(temp_dir, gradcam_filename)
            overlay_bgr = cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_RGB2BGR)
            cv2.imwrite(gradcam_path, overlay_bgr)
            logger.info(f"Grad-CAM saved to {gradcam_path}")
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            gradcam_path = None

    # 4. Save to DB
    pred_id = str(uuid.uuid4())[:8].upper()
    try:
        save_prediction(
            prediction_id=pred_id,
            image_path=image_path,
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time=processing_time,
            model_used=used_model_label
        )
    except Exception as e:
        logger.error(f"Failed to save prediction to DB: {e}")

    return {
        "prediction_id": pred_id,
        "predicted_class": predicted_class,
        "display_name": display_name,
        "confidence": confidence,
        "processing_time": processing_time,
        "model_used": used_model_label,
        "predictions_raw": predictions[0],
        "gradcam_path": gradcam_path,
        "preprocessed_img": preprocessed_img
    }
