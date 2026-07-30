import cv2
import numpy as np
import tensorflow as tf
from config.config import IMG_HEIGHT, IMG_WIDTH
from utils.logger import get_logger

logger = get_logger("Augmentor")

def get_augmentation_pipeline():
    """
    Returns a Keras Sequential augmentation pipeline with a strong set of
    augmentation techniques for improved generalization on field images:
      - Horizontal & vertical flip
      - Random rotation (up to 20°)
      - Random zoom (up to 20%)
      - Random translation (up to 15% shift in x/y)
      - Random brightness and contrast
      - Random hue (TF 2.13+ via lambda)
    """
    logger.info("Initializing enhanced Keras augmentation pipeline...")

    layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.20),       # ±20° rotation
        tf.keras.layers.RandomZoom(0.20),            # ±20% zoom
        tf.keras.layers.RandomTranslation(
            height_factor=0.15, width_factor=0.15   # ±15% shift
        ),
    ]

    # Brightness / contrast
    try:
        layers.append(tf.keras.layers.RandomBrightness(factor=0.25, value_range=(0, 255)))
        layers.append(tf.keras.layers.RandomContrast(factor=0.25))
    except AttributeError:
        logger.warning("RandomBrightness/RandomContrast not available in this Keras version, skipping.")

    return tf.keras.Sequential(layers, name="strong_data_augmentation")


def apply_mixup(images, labels, num_classes, alpha=0.3):
    """
    Apply Mixup augmentation to a batch of images and one-hot encoded labels.
    Blends pairs of training examples to create synthetic training samples.
    Returns mixed images and soft labels.
    """
    batch_size = tf.shape(images)[0]
    lam = tf.cast(
        tf.random.stateless_uniform(
            [], seed=(0, 1), minval=0, maxval=1
        ), tf.float32
    )
    lam = tf.maximum(lam, 1.0 - lam)   # ensure dominant class still dominates

    # Convert labels to one-hot for soft blending
    labels_oh = tf.one_hot(labels, depth=num_classes)

    # Shuffle indices for mix partner
    idx = tf.random.shuffle(tf.range(batch_size))
    images_mix = lam * images + (1.0 - lam) * tf.gather(images, idx)
    labels_mix = lam * labels_oh + (1.0 - lam) * tf.gather(labels_oh, idx)

    return images_mix, labels_mix


def preprocess_and_denoise_image(image_path):
    """
    Reads an image using OpenCV, resizes it to 224×224,
    applies Bilateral Filtering for noise reduction.

    Returns a preprocessed image array in [0, 255] float32 range ready for inference.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Bilateral filter: removes noise while preserving sharp lesion edges
        img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)

        return img_denoised.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise e
