import os
import shutil
import hashlib
import random
from PIL import Image
from config.config import CLASS_NAMES, SPLIT_DATASET_DIR
from utils.logger import get_logger

logger = get_logger("DataPreprocessor")

# Seed for reproducibility
random.seed(42)

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def is_image_valid(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        logger.warning(f"Corrupted image detected and skipped: {file_path}")
        return False

def link_or_copy(src, dst):
    """
    Tries to create a hard link to save disk space and time,
    falls back to copy if linking fails.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)

def prepare_and_split_dataset(src_roots=None,
                             split_root=SPLIT_DATASET_DIR,
                             train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if src_roots is None:
        src_roots = [
            r"d:\Areca\archive\Arecanut_dataset\Arecanut_dataset",
            r"d:\Areca\archive (1)"
        ]
    logger.info(f"Starting dataset preparation from {src_roots} to {split_root}...")

    # Standardize splits to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    # Scanned files index
    all_class_images = {cls: [] for cls in CLASS_NAMES}
    seen_hashes = set()
    duplicate_count = 0
    corrupted_count = 0
    total_scanned = 0

    for root_dir in src_roots:
        if not os.path.exists(root_dir):
            logger.warning(f"Source directory {root_dir} not found, skipping...")
            continue

        # Walk through directory recursively to find matching class folders
        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            matching_class = None
            for cls in CLASS_NAMES:
                if folder_name.lower() == cls.lower():
                    matching_class = cls
                    break

            if matching_class is None:
                continue

            for filename in files:
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mpo')):
                    continue

                file_path = os.path.join(root, filename)
                total_scanned += 1

                # 1. Check corruption
                if not is_image_valid(file_path):
                    corrupted_count += 1
                    continue

                # 2. Check duplicates
                file_hash = calculate_md5(file_path)
                if file_hash and file_hash in seen_hashes:
                    duplicate_count += 1
                    continue

                if file_hash:
                    seen_hashes.add(file_hash)
                all_class_images[matching_class].append(file_path)

    logger.info(f"Scan complete. Total scanned: {total_scanned}")
    logger.info(f"Corrupted images skipped: {corrupted_count}")
    logger.info(f"Duplicate images skipped: {duplicate_count}")

    # Create split distributions
    logger.info("Splitting dataset and creating links/copies...")
    split_summary = {}

    for cls, images in all_class_images.items():
        random.shuffle(images)
        total_images = len(images)
        
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        split_summary[cls] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": total_images
        }
        
        for split_name, split_images in splits.items():
            for idx, src_path in enumerate(split_images):
                ext = os.path.splitext(src_path)[1]
                # Keep filename unique and organized
                dest_filename = f"{cls}_{idx:05d}{ext}"
                dest_path = os.path.join(split_root, split_name, cls, dest_filename)
                link_or_copy(src_path, dest_path)

    logger.info("Dataset preparation and splitting complete.")
    logger.info(f"Split Summary: {split_summary}")
    return split_summary
