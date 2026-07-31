import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import math
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config.config import (
    SPLIT_DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, RESULTS_DIR, SAVED_MODELS_DIR, CLASS_NAMES
)
from preprocessing.data_prep import prepare_and_split_dataset
from augmentation.augmentor import get_augmentation_pipeline
from models.model_registry import MODELS_MAP, get_model_path
from models.architectures import unfreeze_model_top_layers
from utils.logger import get_logger

logger = get_logger("Trainer")

NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_datasets(batch_size=BATCH_SIZE):
    logger.info("Loading train, val, and test datasets from disk...")

    train_dir = os.path.join(SPLIT_DATASET_DIR, "train")
    val_dir   = os.path.join(SPLIT_DATASET_DIR, "val")
    test_dir  = os.path.join(SPLIT_DATASET_DIR, "test")

    if not os.path.exists(train_dir):
        logger.warning("Split dataset not found. Running split preprocessor first...")
        prepare_and_split_dataset()

    common_kwargs = dict(image_size=(IMG_HEIGHT, IMG_WIDTH),
                         batch_size=batch_size,
                         label_mode="int")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, **common_kwargs)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=False, **common_kwargs)
    test_ds  = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, **common_kwargs)

    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
#  Class Weights (handles imbalanced classes)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_class_weights(train_ds):
    logger.info("Calculating class weights for imbalanced dataset handling...")
    labels_list = [labels.numpy() for _, labels in train_ds.unbatch()]
    labels_arr  = np.array(labels_list)
    classes     = np.unique(labels_arr)
    weights     = compute_class_weight(class_weight='balanced',
                                       classes=classes, y=labels_arr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class Weights: {class_weight_dict}")
    return class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
#  Cosine Annealing LR Schedule with Warm Restarts
# ─────────────────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.
    """
    def __init__(self, initial_lr, total_steps, warmup_steps=0, min_lr=1e-7):
        super().__init__()
        self.initial_lr  = float(initial_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr      = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup phase
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        # Cosine decay phase
        progress  = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":  self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr":      self.min_lr,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Function
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model_name,
                epochs=20,
                batch_size=BATCH_SIZE,
                warmup_lr=1e-3,
                finetune_lr=5e-5,
                warmup_epochs=8,
                unfreeze_layers=80,
                label_smoothing=0.1):
    """
    Strong two-stage training pipeline:

    Stage 1 – Head Warmup
        Backbone fully frozen.  Trains the 3-layer classification head
        with Adam + Warmup-Cosine LR decay and Label Smoothing for
        `warmup_epochs` epochs.

    Stage 2 – Backbone Fine-Tuning
        Top `unfreeze_layers` unfrozen (BatchNorm kept frozen).
        AdamW with Cosine LR decay at lower lr=`finetune_lr`.
        EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  STRONG TRAINING: {model_name}")
    logger.info(f"  warmup_epochs={warmup_epochs}  finetune_epochs={epochs}")
    logger.info(f"  warmup_lr={warmup_lr}  finetune_lr={finetune_lr}")
    logger.info(f"  unfreeze_layers={unfreeze_layers}  label_smoothing={label_smoothing}")
    logger.info(f"{'='*60}")

    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured with memory growth.")
        except RuntimeError as e:
            logger.error(f"GPU memory growth error: {e}")

    # ── Load datasets ──
    raw_train_ds, val_ds, test_ds = load_raw_datasets(batch_size=batch_size)
    class_weights = calculate_class_weights(raw_train_ds)

    # Strong augmentation pipeline
    aug_pipeline = get_augmentation_pipeline()
    train_ds = raw_train_ds.map(
        lambda x, y: (aug_pipeline(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Count training batches for LR schedule
    n_train = raw_train_ds.cardinality().numpy()
    if n_train < 0:   # INFINITE or UNKNOWN cardinality fallback
        n_train = sum(1 for _ in raw_train_ds)

    # Dataset pipeline: augment -> prefetch (no huge in-memory shuffle buffer)
    # raw_train_ds was already shuffled=True on load from disk
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    # ── Build model ──
    if model_name not in MODELS_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choices: {list(MODELS_MAP.keys())}")

    model_builder = MODELS_MAP[model_name]
    logger.info(f"Initializing {model_name} architecture (stronger head)...")
    model = model_builder(num_classes=NUM_CLASSES, freeze_base=True)

    # Label-smoothing loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, name="label_smoothed_scce"
    )

    # ══════════════════════════════════════════════════════════════
    #  STAGE 1 — HEAD WARMUP
    # ══════════════════════════════════════════════════════════════
    logger.info(f"--- Stage 1: Head Warmup ({warmup_epochs} epochs) ---")

    warmup_total_steps = warmup_epochs * n_train
    warmup_schedule = WarmupCosineDecay(
        initial_lr=warmup_lr,
        total_steps=warmup_total_steps,
        warmup_steps=max(1, n_train // 2),   # half-epoch linear warmup
        min_lr=1e-6
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights,
        verbose=1
    )

    logger.info(f"Head warmup complete. "
                f"Best val_acc: {max(history_warmup.history['val_accuracy']):.4f}")

    # ══════════════════════════════════════════════════════════════
    #  STAGE 2 — BACKBONE FINE-TUNING
    # ══════════════════════════════════════════════════════════════
    logger.info(f"--- Stage 2: Backbone Fine-Tuning "
                f"({epochs} epochs, top {unfreeze_layers} layers) ---")

    unfreeze_model_top_layers(model, unfreeze_layers=unfreeze_layers)

    finetune_total_steps = epochs * n_train
    finetune_schedule = WarmupCosineDecay(
        initial_lr=finetune_lr,
        total_steps=finetune_total_steps,
        warmup_steps=n_train,   # 1-epoch warmup before cosine decay
        min_lr=1e-8
    )

    # AdamW for fine-tuning (weight decay prevents backbone overfitting)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=finetune_schedule,
            weight_decay=1e-5
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    model_path = get_model_path(model_name)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Optional TensorBoard logging if tensorboard is installed
    try:
        import tensorboard  # noqa: F401
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(RESULTS_DIR, "tensorboard", model_name.replace("-", "_")),
                histogram_freq=1
            )
        )
    except (ImportError, Exception):
        logger.warning("TensorBoard not available in environment; skipping TensorBoard callback.")

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ── Combine & save history ──
    full_history = {}
    for key in history_warmup.history:
        full_history[key] = (history_warmup.history[key]
                             + history_finetune.history.get(key, []))

    clean_name   = model_name.replace('-', '').lower()
    history_file = os.path.join(RESULTS_DIR, f"{clean_name}_history.json")
    with open(history_file, 'w') as f:
        json.dump(full_history, f, indent=2)

    logger.info(f"Training history saved to {history_file}")

    # Final model save (best weights already restored by EarlyStopping)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    best_val = max(full_history.get("val_accuracy", [0]))
    logger.info(f"=== {model_name} BEST VAL ACCURACY: {best_val:.4f} ({best_val*100:.2f}%) ===\n")

    return full_history


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Strong two-stage transfer learning training for ArecaVision AI."
    )
    parser.add_argument("--model", type=str, default="EfficientNet-B0",
                        choices=["EfficientNet-B0", "MobileNetV3", "ResNet50"],
                        help="Model architecture to train.")
    parser.add_argument("--all", action="store_true",
                        help="Train all 3 registered models sequentially.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of fine-tuning epochs (Stage 2).")
    parser.add_argument("--warmup_epochs", type=int, default=8,
                        help="Number of warmup epochs (Stage 1, frozen backbone).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size.")
    parser.add_argument("--warmup_lr", type=float, default=1e-3,
                        help="Peak learning rate during warmup (Stage 1).")
    parser.add_argument("--finetune_lr", type=float, default=5e-5,
                        help="Peak learning rate during fine-tuning (Stage 2).")
    parser.add_argument("--unfreeze_layers", type=int, default=80,
                        help="Number of top backbone layers to unfreeze in Stage 2.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0=off, 0.1 recommended).")

    args = parser.parse_args()

    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_lr=args.warmup_lr,
        finetune_lr=args.finetune_lr,
        warmup_epochs=args.warmup_epochs,
        unfreeze_layers=args.unfreeze_layers,
        label_smoothing=args.label_smoothing,
    )

    if args.all:
        for m in ["EfficientNet-B0", "MobileNetV3", "ResNet50"]:
            train_model(m, **train_kwargs)
    else:
        train_model(args.model, **train_kwargs)
