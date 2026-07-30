import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from config.config import SPLIT_DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, CLASS_NAMES, RESULTS_DIR
from models.model_registry import load_or_create_model
from utils.logger import get_logger

logger = get_logger("Evaluator")

def load_test_dataset():
    test_dir = os.path.join(SPLIT_DATASET_DIR, "test")
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found. Please split dataset first.")
        raise FileNotFoundError(f"Test directory {test_dir} not found.")
        
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False  # Do not shuffle test data to keep labels aligned
    )
    
    return test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def evaluate_model(model_name):
    logger.info(f"Evaluating model: {model_name}...")
    
    test_ds = load_test_dataset()
    model = load_or_create_model(model_name)
    
    # Retrieve true labels and predictions
    y_true = []
    y_pred_probs = []
    
    start_time = time.time()
    for images, labels in test_ds:
        probs = model.predict(images)
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())
    end_time = time.time()
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    total_samples = len(y_true)
    inference_time_ms = ((end_time - start_time) / total_samples) * 1000.0 if total_samples > 0 else 0.0
    
    # Calculate Metrics
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    acc = np.mean(y_true == y_pred)
    
    # Create Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Fill confusion matrix text
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    cm_path = os.path.join(RESULTS_DIR, f"{model_name.replace('-', '').lower()}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Calculate multiclass ROC and Plot
    # Binarize labels for ROC
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    n_classes = len(CLASS_NAMES)
    
    plt.figure(figsize=(10, 8))
    roc_auc_dict = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[CLASS_NAMES[i]] = float(roc_auc)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f"{model_name.replace('-', '').lower()}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"ROC curve saved to {roc_path}")
    
    # Macro metrics calculation
    precisions = [report[c]["precision"] for c in CLASS_NAMES]
    recalls = [report[c]["recall"] for c in CLASS_NAMES]
    f1s = [report[c]["f1-score"] for c in CLASS_NAMES]
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1_score": float(np.mean(f1s)),
        "inference_time_ms": float(inference_time_ms),
        "roc_auc": roc_auc_dict,
        "classification_report": report
    }
    
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name.replace('-', '').lower()}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics

def compare_all_models():
    logger.info("Comparing all saved model metrics...")
    model_names = ["EfficientNet-B0", "MobileNetV3", "ResNet50"]
    comparison = {}
    
    for name in model_names:
        clean_name = name.replace('-', '').lower()
        metrics_path = os.path.join(RESULTS_DIR, f"{clean_name}_metrics.json")
        
        # If metrics don't exist, calculate them (will trigger loading/mocking)
        if not os.path.exists(metrics_path):
            try:
                evaluate_model(name)
            except Exception as e:
                logger.error(f"Could not evaluate model {name}: {e}")
                continue
                
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                comparison[name] = json.load(f)
                
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Comparison report saved to {comparison_path}")
    return comparison

if __name__ == '__main__':
    compare_all_models()
