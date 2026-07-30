import cv2
import numpy as np
import tensorflow as tf
from config.config import IMG_HEIGHT, IMG_WIDTH
from utils.logger import get_logger

logger = get_logger("Augmentor")

def get_augmentation_pipeline():
    """
    Returns a Keras Sequential layer sequence for on-the-fly training augmentation.
    Note: Model-specific preprocessing (e.g. mean subtraction/scaling) is handled inside the model architecture.
    """
    logger.info("Initializing Keras augmentation pipeline...")
    layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
    ]
    
    # RandomBrightness and RandomContrast are available in TF 2.9+ / Keras
    try:
        layers.append(tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0, 255)))
        layers.append(tf.keras.layers.RandomContrast(factor=0.15))
    except AttributeError:
        # Fallback if older TF/Keras version doesn't support them directly in layers
        logger.warning("RandomBrightness/RandomContrast layers not found, skipping brightness/contrast in tf.layers.")
        
    return tf.keras.Sequential(layers, name="data_augmentation")

def preprocess_and_denoise_image(image_path):
    """
    Reads an image using OpenCV, resizes it to 224x224,
    applies Bilateral Filtering for noise reduction.
    
    Returns a preprocessed image array in [0, 255] float32 range ready for inference.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Apply Bilateral Filter (Noise Reduction while keeping edges sharp)
        img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
        
        return img_denoised.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise e
