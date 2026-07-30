"""
dataset_parser.py
-----------------
Parses Roboflow and TensorFlow Object Detection _annotations.csv files
and converts bounding box annotations into standardized formats for training and evaluation.
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional

def parse_annotations_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse a Roboflow/TensorFlow _annotations.csv file.
    Expected columns: filename, width, height, class, xmin, ymin, xmax, ymax
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    # Strip whitespace from column names and string fields
    df.columns = [col.strip() for col in df.columns]
    for col in ['filename', 'class']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Calculate bounding box width, height, and area
    df['box_width'] = df['xmax'] - df['xmin']
    df['box_height'] = df['ymax'] - df['ymin']
    df['box_area'] = df['box_width'] * df['box_height']

    # Relative coordinates [0.0, 1.0]
    df['rel_xmin'] = df['xmin'] / df['width']
    df['rel_ymin'] = df['ymin'] / df['height']
    df['rel_xmax'] = df['xmax'] / df['width']
    df['rel_ymax'] = df['ymax'] / df['height']

    return df


def get_dataset_bounding_boxes(dataset_dir: str, split: str = "train") -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve all annotations grouped by filename for a dataset split.
    Returns dict mapping filename -> list of bounding box annotation dicts.
    """
    csv_path = os.path.join(dataset_dir, split, "_annotations.csv")
    df = parse_annotations_csv(csv_path)

    grouped = {}
    for filename, group in df.groupby('filename'):
        boxes = []
        for _, row in group.iterrows():
            boxes.append({
                "class": row['class'],
                "xmin": int(row['xmin']),
                "ymin": int(row['ymin']),
                "xmax": int(row['xmax']),
                "ymax": int(row['ymax']),
                "width": int(row['width']),
                "height": int(row['height']),
                "rel_xmin": float(row['rel_xmin']),
                "rel_ymin": float(row['rel_ymin']),
                "rel_xmax": float(row['rel_xmax']),
                "rel_ymax": float(row['rel_ymax']),
            })
        grouped[filename] = boxes

    return grouped


def summarize_annotations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics of parsed annotations dataframe.
    """
    class_counts = df['class'].value_counts().to_dict()
    total_boxes = len(df)
    total_images = df['filename'].nunique()

    avg_boxes_per_image = total_boxes / max(total_images, 1)

    return {
        "total_annotations": total_boxes,
        "unique_images": total_images,
        "avg_boxes_per_image": round(avg_boxes_per_image, 2),
        "class_counts": class_counts
    }
