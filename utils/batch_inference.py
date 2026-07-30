"""
batch_inference.py
------------------
Batch inference script for running disease detection on an entire folder
of images without the Streamlit UI. Useful for bulk plantation surveys.

Usage:
    python utils/batch_inference.py --input <folder_path> [--model EfficientNet-B0] [--ensemble]
    python utils/batch_inference.py --input d:/Areca/farm_photos --ensemble

Output:
    - results/batch_results_<timestamp>.csv  — Full per-image result table
    - results/batch_summary_<timestamp>.json — Summary statistics
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CLASS_NAMES, RESULTS_DIR
from utils.inference_helper import run_inference, validate_image
from utils.logger import get_logger

logger = get_logger("BatchInference")

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect_images(input_dir: str) -> list:
    """Recursively finds all valid image files in input_dir."""
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                image_paths.append(full_path)
    return sorted(image_paths)


def run_batch_inference(input_dir: str, model_name: str = "EfficientNet-B0",
                        use_ensemble: bool = False) -> dict:
    """
    Runs inference on all images found in input_dir.

    Args:
        input_dir: Path to directory containing images.
        model_name: Architecture to use (ignored if use_ensemble=True).
        use_ensemble: If True, runs ensemble voting across all 3 models.

    Returns:
        summary: dict with per-image results and aggregate statistics.
    """
    logger.info(f"Starting batch inference on: {input_dir}")
    logger.info(f"Model: {'Ensemble' if use_ensemble else model_name}")

    images = collect_images(input_dir)
    if not images:
        logger.warning(f"No valid images found in {input_dir}")
        return {"total": 0, "results": []}

    logger.info(f"Found {len(images)} images to process.")

    results = []
    errors = []
    class_counts = {cls: 0 for cls in CLASS_NAMES}

    for idx, img_path in enumerate(images, start=1):
        logger.info(f"[{idx}/{len(images)}] Processing: {os.path.basename(img_path)}")

        if not validate_image(img_path):
            logger.warning(f"Skipping corrupted image: {img_path}")
            errors.append({"file": img_path, "error": "Corrupted or invalid image"})
            continue

        try:
            result = run_inference(img_path, model_name=model_name, use_ensemble=use_ensemble)
            class_counts[result["predicted_class"]] = class_counts.get(result["predicted_class"], 0) + 1
            results.append({
                "file": os.path.basename(img_path),
                "full_path": img_path,
                "prediction_id": result["prediction_id"],
                "predicted_class": result["predicted_class"],
                "display_name": result["display_name"],
                "confidence": f"{result['confidence'] * 100:.2f}%",
                "confidence_raw": result["confidence"],
                "processing_time_ms": f"{result['processing_time'] * 1000:.1f}",
                "model_used": result["model_used"],
                "gradcam_path": result["gradcam_path"] or ""
            })
        except Exception as e:
            logger.error(f"Inference failed for {img_path}: {e}")
            errors.append({"file": img_path, "error": str(e)})

    # Build summary
    total = len(results)
    healthy_classes = {"healthy_foot", "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk"}
    healthy_count = sum(1 for r in results if r["predicted_class"] in healthy_classes)
    diseased_count = total - healthy_count

    avg_confidence = (
        sum(r["confidence_raw"] for r in results) / total if total > 0 else 0.0
    )

    top_disease = max(
        {k: v for k, v in class_counts.items() if k not in healthy_classes}.items(),
        key=lambda x: x[1],
        default=("None", 0)
    )[0]

    summary = {
        "batch_dir": input_dir,
        "model": "Ensemble" if use_ensemble else model_name,
        "timestamp": datetime.now().isoformat(),
        "total_images_found": len(images),
        "total_processed": total,
        "total_errors": len(errors),
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "avg_confidence_pct": round(avg_confidence * 100, 2),
        "top_detected_disease": top_disease,
        "class_distribution": class_counts,
        "errors": errors,
        "results": results
    }

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(RESULTS_DIR, f"batch_results_{ts}.csv")
    if results:
        fieldnames = [k for k in results[0].keys() if k != "confidence_raw"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: v for k, v in row.items() if k != "confidence_raw"})
        logger.info(f"Batch CSV results saved to: {csv_path}")

    json_path = os.path.join(RESULTS_DIR, f"batch_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "results"}, f, indent=2)
    logger.info(f"Batch summary saved to: {json_path}")

    logger.info(
        f"Batch complete — {total} processed, {healthy_count} healthy, "
        f"{diseased_count} diseased, {len(errors)} errors."
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ArecaVision AI — Batch Inference Tool"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the folder containing images to process."
    )
    parser.add_argument(
        "--model", type=str, default="EfficientNet-B0",
        choices=["EfficientNet-B0", "MobileNetV3", "ResNet50"],
        help="Model architecture to use (ignored if --ensemble is set)."
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use ensemble voting across all 3 trained models."
    )
    args = parser.parse_args()

    summary = run_batch_inference(
        input_dir=args.input,
        model_name=args.model,
        use_ensemble=args.ensemble
    )

    print("\n=== Batch Inference Summary ===")
    print(f"  Total images processed : {summary['total_processed']}")
    print(f"  Healthy detections     : {summary['healthy_count']}")
    print(f"  Diseased detections    : {summary['diseased_count']}")
    print(f"  Avg confidence         : {summary['avg_confidence_pct']:.2f}%")
    print(f"  Top detected disease   : {summary['top_detected_disease']}")
    print(f"  Errors                 : {summary['total_errors']}")
    print(f"  Results saved to       : {RESULTS_DIR}")
