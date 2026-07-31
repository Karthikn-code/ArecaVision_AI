import os
import numpy as np
import tensorflow as tf
from config.config import SAVED_MODELS_DIR, CLASS_NAMES
from models.architectures import build_efficientnet_b0, build_mobilenet_v3, build_resnet50
from utils.logger import get_logger

try:
    from training.train import WarmupCosineDecay
except ImportError:
    WarmupCosineDecay = None

logger = get_logger("ModelRegistry")

MODELS_MAP = {
    "EfficientNet-B0": build_efficientnet_b0,
    "MobileNetV3": build_mobilenet_v3,
    "ResNet50": build_resnet50
}

def get_model_path(model_name):
    # Sanitize name for filename
    clean_name = model_name.replace("-", "").lower()
    return os.path.join(SAVED_MODELS_DIR, f"{clean_name}.keras")

def load_or_create_model(model_name, num_classes=len(CLASS_NAMES), compile_only=False):
    """
    Tries to load a saved trained model safely. If not found, initializes a fresh
    model with ImageNet weights, compiles it, and returns it.
    """
    import shutil
    import uuid

    model_path = get_model_path(model_name)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    custom_objs = {"WarmupCosineDecay": WarmupCosineDecay} if WarmupCosineDecay else {}

    if os.path.exists(model_path):
        logger.info(f"Loading saved model {model_name} from {model_path}...")
        temp_copy = os.path.join(SAVED_MODELS_DIR, f"temp_{uuid.uuid4().hex[:6]}.keras")
        try:
            shutil.copy2(model_path, temp_copy)
            try:
                # Primary load with compile=False for instant weight restoration without custom schedule deserialization errors
                model = tf.keras.models.load_model(temp_copy, compile=False)
            except Exception:
                model = tf.keras.models.load_model(temp_copy, custom_objects=custom_objs)
            
            # Recompile with standard Adam for metric evaluation & inference
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            logger.info(f"Model {model_name} loaded and compiled successfully.")
            if os.path.exists(temp_copy):
                os.remove(temp_copy)
            return model
        except Exception as e:
            logger.error(f"Error loading saved model copy: {e}. Trying direct load...")
            if os.path.exists(temp_copy):
                try:
                    os.remove(temp_copy)
                except Exception:
                    pass
            try:
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                except Exception:
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                logger.info(f"Direct model load for {model_name} succeeded.")
                return model
            except Exception as direct_e:
                logger.error(f"Direct load failed for {model_name}: {direct_e}.")
            
    # If file does not exist, create a fresh transfer learning model (DO NOT OVERWRITE existing files)
    logger.warning(f"Saved model {model_name} not found. Creating a fresh instance...")
    if model_name not in MODELS_MAP:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODELS_MAP.keys())}")
        
    model_builder = MODELS_MAP[model_name]
    model = model_builder(num_classes=num_classes, freeze_base=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Save fresh instance ONLY if no file exists yet
    if not os.path.exists(model_path):
        try:
            model.save(model_path)
            logger.info(f"Fresh model {model_name} saved to {model_path}.")
        except Exception as e:
            logger.error(f"Failed to save fresh model {model_name}: {e}")
        
    return model

def ensemble_predict(img_batch):
    """
    Runs inference across all 3 trained models and returns the soft-vote
    averaged probability distribution (ensemble prediction).
    
    Args:
        img_batch: numpy array of shape (1, 224, 224, 3) in [0, 255] float32.
    
    Returns:
        averaged_probs: numpy array of shape (1, num_classes)
        loaded_models: dict of {model_name: model} (for Grad-CAM of best model)
    """
    logger.info("Running ensemble prediction across all 3 models...")
    all_probs = []
    loaded_models = {}
    
    for model_name in MODELS_MAP.keys():
        try:
            model = load_or_create_model(model_name)
            probs = model.predict(img_batch, verbose=0)
            all_probs.append(probs)
            loaded_models[model_name] = model
            logger.info(f"Ensemble: {model_name} predicted, max class prob = {probs.max():.3f}")
        except Exception as e:
            logger.error(f"Ensemble: failed to run {model_name}: {e}")
            
    if not all_probs:
        raise RuntimeError("All models failed during ensemble prediction.")
        
    # Soft-vote: average probabilities across all successful models
    averaged_probs = np.mean(all_probs, axis=0)
    logger.info(f"Ensemble averaged prediction complete. Num models used: {len(all_probs)}")
    return averaged_probs, loaded_models

def list_registered_models():
    """
    Returns lists of available model architectures and which ones have been trained/saved.
    """
    status = {}
    for name in MODELS_MAP.keys():
        path = get_model_path(name)
        status[name] = {
            "saved_on_disk": os.path.exists(path),
            "file_path": path
        }
    return status
