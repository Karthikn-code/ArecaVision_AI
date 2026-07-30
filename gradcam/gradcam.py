import numpy as np
import cv2
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger("GradCAM")


def find_last_conv_layer(model):
    """
    Finds the last layer in the model or nested sub-model that outputs a 4D tensor.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                try:
                    if isinstance(sublayer, tf.keras.layers.Conv2D) or len(sublayer.output_shape) == 4:
                        if 'input' not in sublayer.name and 'rescale' not in sublayer.name:
                            return sublayer
                except Exception:
                    pass
        else:
            try:
                if isinstance(layer, tf.keras.layers.Conv2D) or len(layer.output_shape) == 4:
                    if 'input' not in layer.name and 'rescale' not in layer.name:
                        return layer
            except Exception:
                pass
    return None


def compute_gradcam(model, img_array, class_index=None):
    """
    Computes the Grad-CAM heatmap for a given image array and class index.
    Supports nested transfer learning backbones (EfficientNet, MobileNet, ResNet).

    Args:
        model: Compiled Keras model.
        img_array: Preprocessed image array of shape (1, 224, 224, 3).
        class_index: Target class index. If None, uses highest probability class.

    Returns:
        heatmap: 2D numpy heatmap normalized to [0, 1].
    """
    logger.info("Computing Grad-CAM heatmap...")

    # Identify backbone sub-model and target conv layer
    sub_model = None
    target_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            sub_model = layer
            for sublayer in reversed(sub_model.layers):
                try:
                    if isinstance(sublayer, tf.keras.layers.Conv2D) or len(sublayer.output_shape) == 4:
                        if 'input' not in sublayer.name and 'rescale' not in sublayer.name and 'add' not in sublayer.name:
                            target_layer = sublayer
                            break
                except Exception:
                    pass
            if target_layer:
                break
        elif isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break

    if target_layer is None:
        target_layer = find_last_conv_layer(model)

    if target_layer is None:
        raise ValueError("Could not locate a valid 4D Conv layer for Grad-CAM.")

    logger.info(f"Targeting layer for Grad-CAM: '{target_layer.name}' (Sub-model: '{sub_model.name if sub_model else 'None'}')")

    if sub_model is not None:
        # Create intermediate grad model for nested backbone
        sub_grad_model = tf.keras.models.Model(
            inputs=sub_model.input,
            outputs=[target_layer.output, sub_model.output]
        )

        with tf.GradientTape() as tape:
            # Step 1: Preprocess inputs through layers leading up to sub_model
            x = img_array
            for layer in model.layers:
                if layer == sub_model:
                    break
                x = layer(x)

            # Step 2: Forward pass through sub_grad_model
            conv_outputs, sub_out = sub_grad_model(x)
            tape.watch(conv_outputs)

            # Step 3: Forward pass sub_model output through remaining top head layers
            y = sub_out
            sub_found = False
            for layer in model.layers:
                if layer == sub_model:
                    sub_found = True
                    continue
                if sub_found:
                    y = layer(y)

            predictions = y
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
    else:
        # Standard top-level model Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        logger.warning("Gradients evaluated to None. Returning empty heatmap fallback.")
        return np.zeros((224, 224), dtype=np.float32)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weighted sum of feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to isolate positive activation features
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap = heatmap.numpy()
    if np.isnan(heatmap).any():
        heatmap = np.zeros(heatmap.shape, dtype=np.float32)

    return heatmap


def overlay_heatmap(heatmap, original_img, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap on the original RGB image.

    Args:
        heatmap: 2D numpy array normalized to [0, 1].
        original_img: RGB image array of shape (224, 224, 3) in [0, 255] or [0, 1].
        alpha: Heatmap transparency factor.
        colormap: OpenCV colormap constant.

    Returns:
        superimposed_img: Overlay image array in [0, 1] range float32.
    """
    heatmap_255 = np.uint8(255 * heatmap)
    height, width, _ = original_img.shape
    heatmap_resized = cv2.resize(heatmap_255, (width, height))

    # Colorize heatmap (returns BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if original_img.max() <= 1.0:
        orig_255 = np.uint8(255 * original_img)
    else:
        orig_255 = np.uint8(original_img)

    superimposed_img = cv2.addWeighted(heatmap_colored_rgb, alpha, orig_255, 1 - alpha, 0)
    return superimposed_img.astype(np.float32) / 255.0
