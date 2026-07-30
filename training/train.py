import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config.config import (
    SPLIT_DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, RESULTS_DIR, SAVED_MODELS_DIR, CLASS_NAMES
)
from preprocessing.data_prep import prepare_and_split_dataset
from augmentation.augmentor import get_augmentation_pipeline
from models.model_registry import MODELS_MAP, get_model_path
from models.architectures import unfreeze_model_top_layers
from utils.logger import get_logger

logger = get_logger("Trainer")

def load_raw_datasets(batch_size=BATCH_SIZE):
    logger.info("Loading train, val, and test datasets from disk...")
    
    train_dir = os.path.join(SPLIT_DATASET_DIR, "train")
    val_dir = os.path.join(SPLIT_DATASET_DIR, "val")
    test_dir = os.path.join(SPLIT_DATASET_DIR, "test")
    
    # If split dirs do not exist, split the dataset first
    if not os.path.exists(train_dir):
        logger.warning("Split dataset not found. Running split preprocessor first...")
        prepare_and_split_dataset()
        
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    
    return train_ds, val_ds, test_ds

def calculate_class_weights(train_ds):
    logger.info("Calculating class weights for imbalanced dataset handling...")
    labels_list = []
    for _, labels in train_ds.unbatch():
        labels_list.append(labels.numpy())
    labels_list = np.array(labels_list)
    classes = np.unique(labels_list)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels_list)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Calculated Class Weights: {class_weight_dict}")
    return class_weight_dict

def train_model(model_name, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-5, warmup_epochs=5):
    logger.info(f"Starting two-stage training pipeline for {model_name}...")
    
    # Set GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured with memory growth.")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU memory growth: {e}")
            
    # Load Datasets
    raw_train_ds, val_ds, test_ds = load_raw_datasets(batch_size=batch_size)
    
    # Calculate Class Weights
    class_weights = calculate_class_weights(raw_train_ds)
    
    # Apply Data Augmentation to Training Dataset (Images remain float32 [0, 255])
    aug_pipeline = get_augmentation_pipeline()
    train_ds = raw_train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimize dataset pipeline performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Construct fresh architecture instance
    if model_name not in MODELS_MAP:
        raise ValueError(f"Unknown model name: {model_name}. Choices: {list(MODELS_MAP.keys())}")
        
    model_builder = MODELS_MAP[model_name]
    logger.info(f"Initializing {model_name} architecture...")
    model = model_builder(num_classes=len(CLASS_NAMES), freeze_base=True)
    
    # ==================== STAGE 1: WARMUP TOP HEAD ====================
    logger.info(f"--- Stage 1: Warmup Top Head ({warmup_epochs} epochs) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights
    )
    
    # ==================== STAGE 2: BACKBONE FINE-TUNING ====================
    logger.info(f"--- Stage 2: Backbone Fine-Tuning ({epochs} epochs) ---")
    unfreeze_model_top_layers(model, unfreeze_layers=40)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model_path = get_model_path(model_name)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_accuracy", save_best_only=True)
    ]
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Combine History
    full_history = {}
    for key in history_warmup.history.keys():
        full_history[key] = history_warmup.history[key] + history_finetune.history[key]
        
    clean_name = model_name.replace('-', '').lower()
    history_file = os.path.join(RESULTS_DIR, f"{clean_name}_history.json")
    with open(history_file, 'w') as f:
        json.dump(full_history, f, indent=2)
        
    logger.info(f"Saved complete training history to {history_file}")
    
    # Final Model Save
    model.save(model_path)
    logger.info(f"Model saved successfully to {model_path}")
    
    return full_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Areca nut disease classification model.")
    parser.add_argument("--model", type=str, default="EfficientNet-B0",
                        choices=["EfficientNet-B0", "MobileNetV3", "ResNet50"],
                        help="Model architecture to train.")
    parser.add_argument("--all", action="store_true", help="Train all 3 registered models sequentially.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs (Stage 1, frozen backbone).")
    
    args = parser.parse_args()
    
    if args.all:
        for m in ["EfficientNet-B0", "MobileNetV3", "ResNet50"]:
            logger.info(f"\n==================== TRAINING {m} ====================")
            train_model(m, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        warmup_epochs=args.warmup_epochs)
    else:
        train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    warmup_epochs=args.warmup_epochs)

